from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class Attention(nn.Module):
    def __init__(self, idf, cdf, mask: Optional[Tensor] = None):
        super().__init__()
        self._mask = mask
        self.softmax = nn.Softmax()
        self.conv = nn.Conv2d(cdf, idf, kernel_size=1, stride=1, padding=0, bias=False)

    def set_mask(self, mask: Tensor):
        self._mask = mask

    def forward(self, x: Tensor, context: Tensor) -> Tuple[Tensor, Tensor]:
        h, w = x.shape[2:]
        hxw = h * w

        target = x.view(x.shape[0], -1, hxw)
        target_transpose = torch.transpose(target, 1, 2).contiguous()
        source_transpose = self.conv(context.unsqueeze(3)).squeeze(3)

        attn = torch.bmm(target_transpose, source_transpose)
        attn = attn.view(x.shape[0] * hxw, context.shape[2])

        if self._mask is not None:
            mask = self._mask.repeat(hxw, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))

        attn = self.softmax(attn).view(x.shape[0], hxw, context.shape[2])
        attn = torch.transpose(attn, 1, 2).contiguous()

        result = torch.bmm(source_transpose, attn).view(x.shape[0], -1, h, w)
        attn = attn.view(x.shape[0], -1, h, w)

        return result, attn
