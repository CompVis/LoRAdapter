import os
from torch.nn.functional import avg_pool2d, interpolate
from torchvision.transforms.functional import center_crop
import torch

annotator_ckpts_path = os.path.join("~/.cache/custom", "ckpts")


def better_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    ss = imgs.shape
    assert ss[-3] == 3

    H, W = ss[-2:]

    if len(ss) == 3:
        imgs = imgs.unsqueeze(0)

    side = min(H, W)
    factor = side // image_size

    imgs = center_crop(imgs, [side, side])
    if factor > 1:
        imgs = avg_pool2d(imgs, factor)
    imgs = interpolate(imgs, [image_size, image_size], mode="bilinear")

    if len(ss) == 3:
        imgs = imgs[0]
    return imgs
