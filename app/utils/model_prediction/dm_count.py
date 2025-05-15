import torch.nn as nn


# ------------------------------------------------------------------------------
class DMCount(nn.Module):
    """Taken from the orginal implementation of DM-Count (https://github.com/cvlab-stonybrook/DM-Count, https://arxiv.org/pdf/2009.13077.pdf)."""

    def __init__(self):
        super(DMCount, self).__init__()

        # Define the architecture
        self.features = make_vgg_layers()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        # Predict density map mu
        x = self.features(x)
        x = nn.functional.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        mu = self.density_layer(x)

        # Normalize density map
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)

        return mu, mu_normed


# ------------------------------------------------------------------------------
def make_vgg_layers(batch_norm=False):
    """Make layers as specified in the VGG19 architecture."""
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
    ]
    layers = []
    in_channels = 3

    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
