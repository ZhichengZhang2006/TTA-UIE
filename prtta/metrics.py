import torch
from torch import nn
from torch.nn import functional as F


def psnr(prediction, reference):
    mse = (prediction.double() - reference.double()).square().mean(dim=(1, 2, 3))
    return -10 * mse.clamp_min(1e-12).log10()


class SSIM(nn.Module):
    """Per-image RGB SSIM with an 11-pixel Gaussian window and valid borders."""

    def __init__(self):
        super().__init__()
        x = torch.arange(-5, 6, dtype=torch.float32)
        weights = (-x.square() / (2 * 1.5**2)).exp()
        weights /= weights.sum()
        horizontal = weights.view(1, 1, 1, 11).repeat(15, 1, 1, 1)
        self.register_buffer("horizontal", horizontal)
        self.register_buffer("vertical", horizontal.transpose(-2, -1).contiguous())

    def forward(self, prediction, reference):
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x, y = prediction.float(), reference.float()
            moments = torch.cat((x, y, x.square(), y.square(), x * y), dim=1)
            moments = F.conv2d(moments.contiguous(memory_format=torch.channels_last), self.horizontal, groups=15)
            moments = F.conv2d(moments, self.vertical, groups=15)
            mx, my, xx, yy, xy = moments.chunk(5, dim=1)
            vx = (xx - mx.square()).clamp_min(0)
            vy = (yy - my.square()).clamp_min(0)
            covariance = xy - mx * my
            c1, c2 = 0.01**2, 0.03**2
            values = ((2 * mx * my + c1) * (2 * covariance + c2)) / (
                (mx.square() + my.square() + c1) * (vx + vy + c2)
            )
            return values.mean(dim=(1, 2, 3)).clamp(0, 1)
