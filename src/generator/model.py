from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.generator.attention import Attention


class ResidualBlock(nn.Module):
    def __init__(self, n_c: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_c, 2 * n_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_c),
            nn.GLU(dim=1),
            nn.Conv2d(n_c, n_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(n_c)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.model(x)


class MuLogSigmaGetter(nn.Module):
    def __init__(self):
        super().__init__()
        self.glu = nn.GLU(dim=1)
        self.linear = nn.Linear(256, 400)

    def encode(self, sentence_embed: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.glu(self.linear(sentence_embed))
        return x[:, :100], x[:, 100:]

    def forward(self, sentence_embed: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, log_sigma = self.encode(sentence_embed)
        sigma = torch.exp(log_sigma.mul(0.5))
        return torch.randn_like(sigma).mul(sigma).add_(mu), mu, log_sigma


class UpsampleBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c_in, 2 * c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * c_out),
            nn.GLU(dim=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class GeneratorHiddenLayer(nn.Module):
    def __init__(self, ngf: int, nef: int):
        super().__init__()
        self.att = Attention(ngf, nef)

        self.layer = nn.Sequential(
            ResidualBlock(2 * ngf),
            ResidualBlock(2 * ngf),
            UpsampleBlock(2 * ngf, ngf)
        )

    def forward(self, h_code: Tensor, word_embeds: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        self.att.set_mask(mask)
        c_code, att = self.att(h_code, word_embeds)
        return self.layer(torch.cat((h_code, c_code), 1)), att


class Generator(nn.Module):
    def __init__(self, ngf: int = 128, nef: int = 256, ncf: int = 100):
        super().__init__()

        self.gf_dim = 16 * ngf

        self.input_linear = nn.Sequential(
            nn.Linear(100 + ncf, ngf * 16 * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 16 * 4 * 4 * 2),
            nn.GLU(dim=1)
        )

        self.input_upsample = nn.Sequential(
            UpsampleBlock(ngf * 16, ngf * 16 // 2),
            UpsampleBlock(ngf * 16 // 2, ngf * 16 // 4),
            UpsampleBlock(ngf * 16 // 4, ngf * 16 // 8),
            UpsampleBlock(ngf * 16 // 8, ngf * 16 // 16),
        )

        self.get_img_1 = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.h_net2 = GeneratorHiddenLayer(ngf, nef)
        self.get_img_2 = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.h_net3 = GeneratorHiddenLayer(ngf, nef)
        self.get_img_3 = nn.Sequential(
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.mu_log_sigma = MuLogSigmaGetter()

    def forward(self, z_code: Tensor, sent_embed: Tensor,
                word_embeds: Tensor, mask: Tensor) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor]:
        fake_images, att_maps = [], []

        context_code, mu, log_sigma = self.mu_log_sigma(sent_embed)

        code = self.input_linear(torch.cat((z_code, context_code), 1)).view(-1, self.gf_dim, 4, 4)
        h_1 = self.input_upsample(code)

        fake_img1 = self.get_img_1(h_1)
        fake_images.append(fake_img1)

        h_2, att1 = self.h_net2(h_1, word_embeds, mask)
        fake_img2 = self.get_img_2(h_2)
        fake_images.append(fake_img2)

        att_maps.append(att1)

        h_3, att2 = self.h_net3(h_2, word_embeds, mask)
        fake_img3 = self.get_img_3(h_3)
        fake_images.append(fake_img3)

        att_maps.append(att2)

        return fake_images, att_maps, mu, log_sigma
