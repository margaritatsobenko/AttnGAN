import torch
import torch.nn as nn
from torch import Tensor


class ImageEncoder(nn.Module):
    def __init__(self, n_c):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, n_c, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_c, n_c * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_c * 2, n_c * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_c * 4, n_c * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_c * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class DiscriminatorLogits(nn.Module):
    def __init__(self, ndf: int, nef: int, condition: bool = False):
        super().__init__()
        self.ef_dim = nef

        self.conv = None
        if condition:
            self.conv = nn.Sequential(
                nn.Conv2d(8 * ndf + nef, 8 * ndf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(8 * ndf),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid()
        )

    def forward(self, h_code, c_code=None):
        if self.conv is not None and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = self.conv((torch.cat((h_code, c_code), 1)))
            return self.out(h_c_code).view(-1)

        return self.out(h_code).view(-1)


class Discriminator64(nn.Module):
    def __init__(self, ndf: int = 64, nef: int = 256):
        super().__init__()
        self.img_code_s16 = ImageEncoder(ndf)

        self.logits = DiscriminatorLogits(ndf, nef, condition=False)
        self.cond_logits = DiscriminatorLogits(ndf, nef, condition=True)

    def forward(self, x_var):
        return self.img_code_s16(x_var)


class Discriminator128(nn.Module):
    def __init__(self, ndf: int = 64, nef: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            ImageEncoder(ndf),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.logits = DiscriminatorLogits(ndf, nef, condition=False)
        self.cond_logits = DiscriminatorLogits(ndf, nef, condition=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class Discriminator256(nn.Module):
    def __init__(self, ndf: int = 64, nef: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            ImageEncoder(ndf),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 32, ndf * 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 16, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.logits = DiscriminatorLogits(ndf, nef, condition=False)
        self.cond_logits = DiscriminatorLogits(ndf, nef, condition=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
