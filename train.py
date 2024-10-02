import hydra
import math
from src.model import ModelBase
from diffusers.optimization import get_scheduler
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from accelerate.logging import get_logger
import signal
import einops
import os
import traceback
from functools import reduce

from src.utils import add_lora_from_config, save_checkpoint


torch.set_float32_matmul_precision("high")


stop_training = False


def signal_handler(sig, frame):
    global stop_training
    stop_training = True
    print("got stop signal")


@hydra.main(config_path="configs", config_name="train")
def main(cfg):
    signal.signal(signal.SIGUSR1, signal_handler)
    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
        log_with="tensorboard",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    logger = get_logger(__name__)

    logger.info("==================================")
    logger.info(cfg)
    logger.info(output_path)

    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model

    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device)
    n_loras = len(cfg.lora.keys())

    cfg_mask = add_lora_from_config(model, cfg, accelerator.device)

    if cfg.get("gradient_checkpointing", False):
        model.unet.enable_gradient_checkpointing()

    dm = cfg.data

    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    mappers_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, []))
    )
    encoder_params = list(
        filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, []))
    )

    optimizer = torch.optim.AdamW(
        model.params_to_optimize + mappers_params + encoder_params,
        lr=cfg.learning_rate,
    )

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
    )

    logger.info(f"Number params Mapper Network(s) {sum(p.numel() for p in mappers_params):,}")
    logger.info(f"Number params Encoder Network(s) {sum(p.numel() for p in encoder_params):,}")
    logger.info(f"Number params all LoRAs(s) {sum(p.numel() for p in model.params_to_optimize):,}")

    logger.info("init trackers")
    if accelerator.is_main_process:
        accelerator.init_trackers("tensorboard")

    logger.info("prepare network")

    prepared = accelerator.prepare(
        *model.mappers,
        *model.encoders,
        model.unet,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    mappers = prepared[: len(model.mappers)]
    encoders = prepared[len(model.mappers) : len(model.mappers) + len(model.encoders)]
    (unet, optimizer, train_dataloader, val_dataloader, lr_scheduler) = prepared[
        len(model.mappers) + len(model.encoders) :
    ]
    model.unet = unet
    model.mappers = mappers
    model.encoders = encoders

    try:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)

        if cfg.get("max_train_steps", None) is None:
            max_train_steps = cfg.epochs * num_update_steps_per_epoch
        else:
            max_train_steps = cfg.max_train_steps
    except:
        max_train_steps = 10000000

    global_step = 0
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_main_process,
    )
    progress_bar.set_description("Steps")

    logger.info("start training")
    for epoch in range(cfg.epochs):
        logger.info("new epoch")
        unet.train()
        map(lambda m: m.train(), mappers)
        map(lambda m: m.train(), encoders)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, *mappers, *encoders):
                imgs = batch["jpg"]
                imgs = imgs.to(accelerator.device)
                imgs = imgs.clip(-1.0, 1.0)
                B = imgs.shape[0]

                cs = [imgs] * n_loras

                if cfg.get("prompt", None) is not None:
                    prompts = [cfg.prompt] * B
                else:
                    prompts = batch["caption"]

                # cfg mask to always true such that the model always learns dropout
                model_pred, loss, x0, _ = model.forward_easy(
                    imgs,
                    prompts,
                    cs,
                    cfg_mask=[True for _ in cfg_mask],
                    batch=batch,
                )

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs, refresh=False)
            accelerator.log(logs, step=global_step)

            # after every gradient update step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % cfg.val_steps != 0 and not stop_training:
                    continue

                # VALIDATION
                with torch.no_grad():
                    try:
                        unet.eval()
                        map(lambda m: m.eval(), mappers)
                        map(lambda m: m.eval(), encoders)

                        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                        val_prompts = []
                        for i, val_batch in enumerate(val_dataloader):

                            B = val_batch["jpg"].shape[0]

                            if i >= cfg.get("val_batches", 4):
                                break

                            if cfg.get("prompt", None) is not None:
                                prompts = [cfg.prompt] * B
                            else:
                                prompts = val_batch["caption"]

                            val_prompts = prompts

                            imgs = val_batch["jpg"]
                            imgs = imgs.to(accelerator.device)
                            imgs = imgs.clip(-1.0, 1.0)

                            cs = [imgs] * n_loras

                            pipeline_args = {
                                "prompt": prompts,
                                "num_images_per_prompt": 1,
                                "cs": cs,
                                "generator": generator,
                                "cfg_mask": cfg_mask,
                                "batch": val_batch,
                            }

                            preds = model.sample(**pipeline_args)

                        if accelerator.is_main_process:
                            # IMAGE saving
                            if cfg.get("log_c", False):
                                # ALWAYS in [0, 1]
                                lp = model.encoders[0](cs[-1]).cpu()
                            else:
                                lp = (imgs.cpu() + 1) / 2

                            lp = torch.nn.functional.interpolate(
                                lp,
                                size=(cfg.size, cfg.size),
                                mode="bicubic",
                                align_corners=False,
                            )

                            log_cond = TF.to_pil_image(einops.rearrange(lp, "b c h w -> c h (b w) "))
                            log_cond = log_cond.convert("RGB")
                            log_cond = np.asarray(log_cond)

                            log_pred = np.concatenate(
                                [np.asarray(img.resize((cfg.size, cfg.size))) for img in preds],
                                axis=1,
                            )

                            for tracker in accelerator.trackers:
                                if tracker.name == "tensorboard":
                                    np_images = np.concatenate([log_cond, log_pred], axis=0)
                                    tracker.writer.add_images(
                                        "validation",
                                        np_images,
                                        global_step,
                                        dataformats="HWC",
                                    )
                                    tracker.writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], global_step)
                                    tracker.writer.add_scalar("loss", loss.detach().item(), global_step)
                                    tracker.writer.add_text(
                                        "prompts",
                                        "------------".join(val_prompts),
                                        global_step,
                                    )

                    except Exception as e:
                        print("!!!!!!!!!!!!!!!!!!!")
                        print("ERROR IN VALIDATION")
                        print(e)
                        print(traceback.format_exc())
                        print("!!!!!!!!!!!!!!!!!!!")

                    finally:
                        if accelerator.is_main_process:
                            save_checkpoint(
                                model.get_lora_state_dict(accelerator.unwrap_model(unet)),
                                [accelerator.unwrap_model(m).state_dict() for m in mappers],
                                None,
                                output_path / f"checkpoint-{global_step}",
                            )

                        unet.train()
                        map(lambda m: m.train(), mappers)
                        map(lambda m: m.train(), encoders)

            if stop_training:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(
        model.get_lora_state_dict(accelerator.unwrap_model(unet)),
        [accelerator.unwrap_model(m).state_dict() for m in mappers],
        None,
        output_path / f"checkpoint-{global_step}",
    )


if __name__ == "__main__":
    main()
