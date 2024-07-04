from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F


class SquarePad(Transform):
    # use standard pad transform of v2
    # but always pads it to be a square

    def __init__(self):
        super().__init__()

        # self.fill = fill
        # self.padding_mode = padding_mode

    def _transform(self, inpt, params):
        h, w = inpt.shape[-2], inpt.shape[-1]

        if h > w:
            padding = [h - w // 2, 0, h - w // 2, 0]
        else:
            padding = [0, w - h // 2, 0, w - h // 2]

        return F.pad(inpt, padding, fill=255)


class TopCrop(Transform):
    # use standard crop transform of v2
    # but always crops from the top

    def __init__(self, size):
        super().__init__()
        self.size = size

    def _transform(self, inpt, params):
        return F.crop(inpt, 0, 0, self.size, self.size)
