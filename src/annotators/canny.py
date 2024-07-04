import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np


class CannyDetector:

    def to(self, *args, **kwargs):
        return self

    def __call__(self, imgs, low_threshold=100, high_threshold=250):
        assert isinstance(imgs, torch.Tensor)
        assert imgs.ndim == 4
        assert imgs.shape[1] == 3
        assert imgs.dtype == torch.float32
        assert imgs.max() <= 1.0
        assert imgs.min() >= -1.0

        imgs = (imgs + 1.0) / 2.0
        edges = []
        for img in imgs:
            img = TF.to_pil_image(img)
            img = np.array(img)
            edge = cv2.Canny(img, low_threshold, high_threshold)
            edge = TF.to_tensor(edge)
            edge = edge.repeat_interleave(3, dim=0)
            edges.append(edge)

        return torch.stack(edges).to(imgs.device)
