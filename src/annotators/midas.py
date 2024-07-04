import torch
from torch import nn
from jaxtyping import Float
from torchvision.transforms.functional import resize

from transformers import (
    DPTImageProcessor,
    DPTForDepthEstimation,
)

from .util import better_resize


class DepthEstimator(nn.Module):

    def __init__(
        self,
        size: int,
        model: str = "Intel/dpt-hybrid-midas",
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.size = size
        self.model_size = 384

        self.depth_estimator = DPTForDepthEstimation.from_pretrained(model, local_files_only=local_files_only)
        # self.feature_extractor = DPTImageProcessor.from_pretrained(model, local_files_only=local_files_only)

        self.depth_estimator.requires_grad_(False)
        self.depth_estimator.eval()

    @torch.no_grad()
    def forward(
        self,
        imgs: Float[torch.Tensor, "B C H W"],
    ) -> Float[torch.Tensor, "B C H W"]:
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        imgs = (imgs + 1.0) / 2.0
        imgs = better_resize(imgs, self.model_size)
        # depth_dict = self.feature_extractor(imgs, do_rescale=False, return_tensors="pt")

        # for k, v in depth_dict.items():
        #     if isinstance(v, torch.Tensor):
        #         depth_dict[k] = v.to(device=imgs.device)

        depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(self.size, self.size),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        depth_map = torch.cat([depth_map] * 3, dim=1)

        # in [0.0, 1.0]
        return depth_map
