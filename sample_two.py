import hydra
import math

from src.utils import DataProvider
from src.model import ModelBase
from diffusers.optimization import get_scheduler
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
from torch import nn
from pathlib import Path
import numpy as np
import torchvision.transforms.functional as TF
from accelerate.logging import get_logger
from PIL import Image
from functools import reduce

from src.utils import add_lora_from_config


# only used this for HED so far
torch.set_float32_matmul_precision("high")


def get_imgs_from_batch(batch: dict[str, torch.Tensor], is_video=False) -> torch.Tensor:
    if is_video:
        B, C, T, H, W = batch["sequence"].shape

        batch_selector = torch.linspace(0, B - 1, B, dtype=torch.int)
        frame_selector = torch.randint(0, T, (B,))

        # imgs in [-1, 1]
        imgs = batch["sequence"]
        imgs = imgs[batch_selector, :, frame_selector]
        return imgs

    imgs = batch["jpg"]

    return imgs


@hydra.main(config_path="configs", config_name="sample")
def main(cfg):
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accelerator = Accelerator(
        project_dir=output_path / "logs",
    )

    str_cfg = cfg
    print(str_cfg)
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model

    model = model.to(accelerator.device)
    model.pipe.to(accelerator.device)

    weight_type = torch.float32
    if cfg.get("bf16", False):
        weight_type = torch.bfloat16

    cfg_mask = add_lora_from_config(model, cfg, accelerator.device, weight_type)

    model.unet.to(accelerator.device, weight_type)
    model = model.to(accelerator.device, weight_type)
    model.pipe.to(accelerator.device, weight_type)

    print(cfg_mask)

    dm1 = cfg.data
    val_dataloader1 = dm1.val_dataloader()
    print(val_dataloader1)

    val_dataloader2 = dm1.val_dataloader()
    try:
        dm2 = cfg.data2
        val_dataloader2 = dm2.val_dataloader()
    except:
        print("no second dataloader provided")

    logger = get_logger(__name__)

    logger.info("==================================")
    logger.info(str_cfg)
    logger.info(output_path)

    logger.info("prepare network")
    val_dataloader1 = accelerator.prepare(val_dataloader1)
    unet = model.unet
    # model.unet = unet

    unet.requires_grad_(False)
    unet.eval()

    images = []
    val_prompts = []
    for it, val_batch in enumerate(tqdm(val_dataloader1)):

        if it < cfg.get("skip", 0):
            continue

        for ib, val_batch2 in enumerate(tqdm(val_dataloader2)):

            generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

            i = max(it, ib)

            if cfg.get("prompt", None) is not None:
                if len(cfg.prompt) > 1:
                    prompts = cfg.prompt
                else:
                    prompts = [cfg.prompt]
            else:
                prompts = val_batch["caption"]

            print(prompts)
            val_prompts.append(prompts)

            # B, C, T, H, W = batch["sequence"].shape
            # imgs = get_imgs_from_batch(val_batch, cfg.get("is_video", False))

            imgs = val_batch["jpg"]
            imgs = imgs.to(accelerator.device, weight_type)
            imgs = imgs.clip(-1.0, 1.0)

            imgs2 = val_batch2["jpg"]
            imgs2 = imgs2.to(accelerator.device, weight_type)
            imgs2 = imgs2.clip(-1.0, 1.0)

            cs = [imgs, imgs2]

            pipeline_args = {
                "prompt": prompts,
                "num_images_per_prompt": cfg.n_samples,
                "cs": cs,
                "generator": generator,
                "cfg_mask": cfg_mask,
                # "prompt_offset_step": cfg.get("prompt_offset_step", 0),
            }

            preds = model.sample(**pipeline_args)

            for j, pred in enumerate(preds):
                pred.save(f"{accelerator.process_index}-img_{it}_{ib}_{j}_sample.jpg")

            if cfg.get("save_grid", False):

                if cfg.get("log_cond", False):
                    # depth is in [0, 1]
                    cond1 = (imgs + 1) / 2
                    cond2 = model.encoders[-1](imgs2)
                    log_pils = [TF.to_pil_image((torch.cat([c1, c2], dim=2)).float().cpu()) for c1, c2 in zip(cond1, cond2)]
                else:
                    log_pils = [TF.to_pil_image((img.float().cpu() + 1) / 2) for img in imgs]

                for j, log_pil in enumerate(log_pils):
                    log_pil.save(f"{accelerator.process_index}-img_{i}_{j}_prompt.jpg")

                images.append(
                    np.concatenate(
                        # we know height is constant
                        [np.asarray(img.resize((int(cfg.size * img.width / img.height), cfg.size))) for img in [*log_pils, *preds]],
                        axis=1,
                    )
                )

    if cfg.get("save_grid", False):
        np_images = np.concatenate(images, axis=0)
        Image.fromarray(np_images).save("test.jpg")


if __name__ == "__main__":
    main()
