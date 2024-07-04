import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms as T
from src.annotators.util import annotator_ckpts_path

# Taken from https://github.com/lllyasviel/ControlNet/blob/main/annotator/hed/__init__.py
# Thanks


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(
            torch.nn.Conv2d(
                in_channels=input_channel,
                out_channels=output_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,
            )
        )
        for i in range(1, layer_number):
            self.convs.append(
                torch.nn.Conv2d(
                    in_channels=output_channel,
                    out_channels=output_channel,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=1,
                )
            )
        self.projection = torch.nn.Conv2d(
            in_channels=output_channel,
            out_channels=1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
        )

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def forward(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


class TorchHEDdetector(nn.Module):
    def __init__(self, size, return_without_channels: bool = False, local_files_only: bool = False):
        super().__init__()

        self.size = size
        self.return_without_channels = return_without_channels

        remote_model_path = "https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth"
        modelpath = os.path.join(annotator_ckpts_path, "ControlNetHED.pth")
        if not os.path.exists(modelpath):
            from basicsr.utils.download_util import load_file_from_url

            load_file_from_url(remote_model_path, model_dir=annotator_ckpts_path)

        self.netNetwork = ControlNetHED_Apache2().float().eval()
        self.netNetwork.load_state_dict(torch.load(modelpath))

        self.netNetwork = self.netNetwork.eval()
        self.netNetwork.requires_grad_(False)

    # converts imgs from [-1, 1] to [0, 255]
    # returns back in [0, 1]
    # @torch.no_grad()
    def forward(self, image_hed, return_without_channels: bool = False):
        assert isinstance(image_hed, torch.Tensor)
        assert image_hed.ndim == 4
        assert image_hed.shape[1] == 3
        assert image_hed.dtype == torch.float32
        assert image_hed.max() <= 1.0
        assert image_hed.min() >= -1.0

        resize = T.Resize((self.size, self.size), T.InterpolationMode.BICUBIC)

        # yes it's supposed to be in [0, 255] as float32
        image_hed = (image_hed + 1.0) * 127.5

        edges = self.netNetwork(image_hed)
        edges = [e[:, 0] for e in edges]
        edges = [resize(e) for e in edges]
        edges = torch.stack(edges, dim=3)
        edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=3)))

        if return_without_channels or self.return_without_channels:
            return edge

        edge = edge[:, None, :, :].repeat_interleave(3, 1)
        return edge
