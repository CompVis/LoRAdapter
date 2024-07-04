import torch
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


class VisionModel(nn.Module):

    def __init__(
        self,
        clip_model: str,
        with_projection: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.clip_model = clip_model
        self.with_projection = with_projection
        if with_projection:
            self.clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model, local_files_only=local_files_only)
        else:
            self.clip_vision_model = CLIPVisionModel.from_pretrained(clip_model, local_files_only=local_files_only)
        self.clip_vision_processor = CLIPImageProcessor.from_pretrained(clip_model, local_files_only=local_files_only)

        self.clip_vision_model.requires_grad_(False)
        self.clip_vision_model.eval()

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        assert imgs.min() >= -1.0
        assert imgs.max() <= 1.0
        assert len(imgs.shape) == 4

        imgs = (imgs + 1) * 127.5
        imgs = imgs.to(dtype=torch.uint8)
        clip_vision_inputs = self.clip_vision_processor(images=imgs, return_tensors="pt")

        for k, v in clip_vision_inputs.items():
            if isinstance(v, torch.Tensor):
                clip_vision_inputs[k] = v.to(device=imgs.device)

        vision_outputs = self.clip_vision_model(**clip_vision_inputs)
        last_hidden_state = vision_outputs.last_hidden_state
        if self.with_projection:
            out = vision_outputs.image_embeds
        else:
            out = vision_outputs.pooler_output

        return out
