import torch
from torch import nn
from torch.nn import functional as F


def conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.GroupNorm(8, out_channels),
        nn.SiLU(inplace=True),
    )


class AdapterProjection(nn.Module):
    def __init__(self, in_channels, out_channels, rank, alpha):
        super().__init__()
        self.base = nn.Conv2d(in_channels, out_channels, 1)
        self.down = nn.Conv2d(in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, out_channels, 1, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_normal_(self.down.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return self.base(features) + self.scale * self.up(self.down(features))


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, adapter):
        super().__init__()
        merged_channels = in_channels + out_channels
        if adapter is None:
            self.project = nn.Identity()
        else:
            self.project = AdapterProjection(merged_channels, out_channels, **adapter)
            merged_channels = out_channels
        self.refine = nn.Sequential(conv(merged_channels, out_channels), conv(out_channels, out_channels))

    def forward(self, features, skip):
        up = F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.project(torch.cat((up, skip), dim=1)))


class UNet4(nn.Module):
    def __init__(self, channels, rank, alpha, positions):
        super().__init__()
        if len(channels) != 4 or not set(positions) <= {f"decoder_stage{i}" for i in range(1, 4)}:
            raise ValueError("UNet4 needs four widths and adapter positions within its three decoder stages")
        self.encoder = nn.ModuleList(
            nn.Sequential(conv(source, width, 1 if i == 0 else 2), conv(width, width))
            for i, (source, width) in enumerate(zip([3, *channels[:-1]], channels))
        )
        self.decoder = nn.ModuleList(
            Decoder(source, width, dict(rank=rank, alpha=alpha) if f"decoder_stage{i}" in positions else None)
            for i, (source, width) in enumerate(zip(channels[:0:-1], channels[-2::-1]), start=1)
        )
        self.output = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, image):
        skips = []
        for encode in self.encoder:
            image = encode(image)
            skips.append(image)
        for decode, skip in zip(self.decoder, skips[-2::-1]):
            image = decode(image, skip)
        return self.output(image).sigmoid()

    def adapter_parameters(self):
        for stage in self.decoder:
            if isinstance(stage.project, AdapterProjection):
                yield stage.project.down.weight
                yield stage.project.up.weight

    def freeze_for_adaptation(self):
        self.eval().requires_grad_(False)
        for parameter in self.adapter_parameters():
            parameter.requires_grad_(True)


def build_model(config, device):
    static = config["static"]
    adapter = static["adapter"]
    return UNet4(static["channels"], adapter["rank"], adapter["alpha"], adapter["positions"]).to(
        device=device, memory_format=torch.channels_last,
    )
