from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.encoders.image_encoder import CNNEncoder


def attention(x: Tensor, context: Tensor, gamma: float) -> Tuple[Tensor, Tensor]:
    bs, x_l = x.shape[0], x.shape[2]
    h, w = context.shape[2:]
    hxw = h * w

    context = context.view(bs, -1, hxw)
    context_transpose = torch.transpose(context, 1, 2).contiguous()

    attn = F.softmax(torch.bmm(context_transpose, x).view(bs * hxw, x_l))
    attn = attn.view(bs, hxw, x_l)
    attn = torch.transpose(attn, 1, 2).contiguous().view(bs * x_l, hxw)

    attn = F.softmax(attn * gamma).view(bs, x_l, hxw)

    attn_transpose = torch.transpose(attn, 1, 2).contiguous()
    result = torch.bmm(context, attn_transpose)

    return result, attn_transpose.view(bs, -1, h, w)


class Loss:
    def __init__(self, device):
        self.device = device
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def sent_loss(self, cnn_code: Tensor, sent_embeds: Tensor,
                  labels: Tensor, class_ids: np.ndarray) -> Tensor:
        masks = []
        for i in range(sent_embeds.shape[0]):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks).to(self.device)

        if cnn_code.dim() == 2:
            cnn_code = cnn_code.unsqueeze(0)
            sent_embeds = sent_embeds.unsqueeze(0)

        cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
        sent_embeds_norm = torch.norm(sent_embeds, 2, dim=2, keepdim=True)

        scores0 = torch.bmm(cnn_code, sent_embeds.transpose(1, 2))
        norm0 = torch.bmm(cnn_code_norm, sent_embeds_norm.transpose(1, 2))

        scores0 = (scores0 / norm0.clamp(min=1e-8) * 10.0).squeeze()
        scores0.data.masked_fill_(masks, -float('inf'))

        scores1 = scores0.transpose(0, 1)

        return self.ce_loss(scores0, labels) + self.ce_loss(scores1, labels)

    def words_loss(self, img_features: Tensor, words_embeds: Tensor, labels: Tensor,
                   cap_lens: Tensor, class_ids: np.ndarray) -> Tensor:
        bs = words_embeds.shape[0]
        cap_lens = cap_lens.data.tolist()

        masks, similarities = [], []
        for i in range(bs):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

            words_num = cap_lens[i]
            word = words_embeds[i, :, :words_num].unsqueeze(0).contiguous()
            word = word.repeat(bs, 1, 1)

            w_cont, attn = attention(word, img_features, 5.0)
            w_cont = w_cont.transpose(1, 2).contiguous().view(bs * words_num, -1)

            word = word.transpose(1, 2).contiguous().view(bs * words_num, -1)
            row_sim = torch.cosine_similarity(word, w_cont).view(bs, words_num)
            similarities.append(torch.log(torch.exp(5 * row_sim).sum(dim=1, keepdim=True)))

        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks).to(self.device)

        similarities = 10.0 * torch.cat(similarities, 1)
        similarities.data.masked_fill_(masks, -float('inf'))
        similarities1 = similarities.transpose(0, 1)

        return self.ce_loss(similarities, labels) + self.ce_loss(similarities1, labels)

    def discriminator_loss(self, discriminator, real_images: Tensor, fake_images: Tensor,
                           conditions, real_labels, fake_labels) -> Tensor:
        real_features = discriminator(real_images)
        fake_features = discriminator(fake_images.detach())

        cond_real_logits = discriminator.cond_logits(real_features, conditions)
        cond_real_bce_loss = self.bce_loss(cond_real_logits, real_labels)
        cond_fake_logits = discriminator.cond_logits(fake_features, conditions)
        cond_fake_bce_loss = self.bce_loss(cond_fake_logits, fake_labels)

        bs = real_features.shape[0]
        cond_wrong_logits = discriminator.cond_logits(real_features[:(bs - 1)], conditions[1:bs])
        cond_wrong_bce_loss = self.bce_loss(cond_wrong_logits, fake_labels[1:bs])

        real_logits = discriminator.logits(real_features)
        fake_logits = discriminator.logits(fake_features)

        real_bce_loss = self.bce_loss(real_logits, real_labels)
        fake_bce_loss = self.bce_loss(fake_logits, fake_labels)

        real_loss = 0.5 * (real_bce_loss + cond_real_bce_loss)

        return real_loss + (fake_bce_loss + cond_fake_bce_loss + cond_wrong_bce_loss) / 3.

    def generator_loss(self, discriminators, image_encoder: CNNEncoder, fake_images: Tensor, real_labels,
                       words_embeds, sent_embeds, match_labels, cap_lens, class_ids) -> Tensor:
        g_total_loss = 0
        for i in range(3):
            features = discriminators[i](fake_images[i])
            cond_logits = discriminators[i].cond_logits(features, sent_embeds)
            cond_bce_loss = self.bce_loss(cond_logits, real_labels)
            logits = discriminators[i].logits(features)
            g_total_loss += self.bce_loss(logits, real_labels) + cond_bce_loss

        region_features, cnn_code = image_encoder(fake_images[2])

        w_loss = self.words_loss(region_features, words_embeds, match_labels, cap_lens, class_ids)
        s_loss = self.sent_loss(cnn_code, sent_embeds, match_labels, class_ids)

        g_total_loss += w_loss + s_loss

        return g_total_loss

    @staticmethod
    def kl_loss(mu: Tensor, log_sigma: Tensor) -> Tensor:
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        temp = mu.pow(2).add_(log_sigma.exp()).mul_(-1).add_(1).add_(log_sigma)
        return torch.mean(temp).mul_(-0.5)
