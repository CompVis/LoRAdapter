from typing import Any, Literal
import torch
from accelerate import Accelerator
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from functools import reduce
import os

# from src.model import ModelBase

MODE = Literal[
    "train",
    "val",
    "always",
]


class DataProvider:
    def __init__(self):
        self.batch = None

    def set_batch(self, batch):
        if self.batch is not None:
            if isinstance(self.batch, torch.Tensor):
                assert self.batch.shape[1:] == batch.shape[1:], "Check: shapes probably should not change during training"

        self.batch = batch

    def get_batch(self):
        assert self.batch is not None, "Error: need to set a batch first"

        return self.batch

    def reset(self):
        self.batch = None


def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def add_lora_from_config(model, cfg: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> list[bool]:
    total_dict_keys: list[str] = []
    cfg_mask: list[bool] = []

    global_ckpt_path = cfg.get("ckpt_path", None)
    project_root = Path(os.path.abspath(__file__)).parent.parent

    for name, l in cfg.lora.items():
        if l.get("enable", "always") == "never":
            continue

        optimize = l.get("optimize", False)
        lora_cfg = l.config
        print(f"Adding {name} lora! Optimize: {optimize}")

        dp = DataProvider()
        mapper_network = l.mapper_network.to(device, dtype)
        encoder = l.encoder.to(device, dtype)
        local_ckpt_path = l.get("ckpt_path", None)

        model.add_lora_to_unet(
            lora_cfg,
            name=name,
            data_provider=dp,
            mapper=mapper_network,
            encoder=encoder,
            optimize=optimize,
            transforms=l.get("transforms", []),
        )

        cfg_mask.append(l.get("cfg", True))

        p = None
        if global_ckpt_path is not None:
            p = Path(project_root, global_ckpt_path) / name

        # local checkpoints path always override global ones
        if local_ckpt_path is not None:
            p = Path(project_root, local_ckpt_path) / name

        if p is not None:
            print("loaded checkpoint for lora", name)
            mapper_sd = torch.load(p / "mapper-checkpoint.pt", map_location=device)
            lora_sd = torch.load(p / "lora-checkpoint.pt", map_location=device)

            if os.path.isfile(p / "encoder-checkpoint.pt"):
                encoder_sd = torch.load(p / "encoder-checkpoint.pt", map_location=device)
                encoder.load_state_dict(encoder_sd)

            mapper_network.load_state_dict(mapper_sd)

            if not optimize:
                mapper_network.requires_grad_(False)
                mapper_network.eval()

            model.unet.load_state_dict(lora_sd, strict=False)
            model.unet.to(device, dtype)
            total_dict_keys += list(lora_sd.keys())

    if len(total_dict_keys) > 0 and not cfg.get("ignore_check", False):
        assert set([v for vs in model.lora_state_dict_keys.values() for v in vs]) == set(
            total_dict_keys
        ), "Probably missing or incorrect checkpoint file path. Otherwise set ignore_check=true in config."

    return cfg_mask


def toggle_loras(model, cfg: Any, mode: MODE):
    for name, l in cfg.lora.items():
        if l.get("enable", "always") in [mode, "always"]:
            for layer in model.lora_layers[name]:
                layer.lora_scale = l.config.get("lora_scale", 1.0)
        else:
            try:
                for layer in model.lora_layers[name]:
                    layer.lora_scale = 0.0
            except:
                print(f"LoRA {name} is disabled. Ignoring...")


def global_gradient_norm(model):
    mappers_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, [])))
    encoder_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, [])))

    total_norm = clip_grad_norm_(model.params_to_optimize + mappers_params + encoder_params, 1e9)
    return total_norm.item()


def save_checkpoint(unet_sds: dict[str, dict[str, torch.Tensor]], mapper_network_sd: list[dict[str, torch.Tensor]], encoder_sd: list[dict[str, torch.Tensor]] | None, path: Path):
    for i, (name, sd) in enumerate(unet_sds.items()):
        p = path / name
        p.mkdir(parents=True, exist_ok=True)

        torch.save(sd, p / "lora-checkpoint.pt")
        torch.save(mapper_network_sd[i], p / f"mapper-checkpoint.pt")
        if encoder_sd is not None and len(encoder_sd[i]) > 0:
            torch.save(encoder_sd[i], p / f"encoder-checkpoint.pt")


def roll_list(l, n):
    # consistent with torch.roll
    return l[-n:] + l[:-n]
