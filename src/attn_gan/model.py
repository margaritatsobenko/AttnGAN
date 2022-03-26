import os
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.attn_gan.losses import Loss
from src.discriminator.model import Discriminator64, Discriminator128, Discriminator256
from src.encoders.image_encoder import CNNEncoder
from src.encoders.text_encoder import RNNEncoder
from src.generator.model import Generator
from src.objects.utils import prepare_data


class AttnGANRunner(object):
    def __init__(self, data_loader: DataLoader, n_words: int, code2word: Dict[int, str],
                 image_dir: str, cnn_weights: str, rnn_weights: str,
                 generator_path_save: str, discriminators_path_save: str,
                 num_epochs: int = 45, noise_dim: int = 100):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.n_words = n_words
        self.code2word = code2word
        self.noise_dim = noise_dim
        self.num_epochs = num_epochs
        self.loss = Loss(self.device)
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size

        self.image_dir = image_dir
        self.cnn_weights = cnn_weights
        self.rnn_weights = rnn_weights
        self.generator_path_save = generator_path_save
        self.discriminators_path_save = discriminators_path_save

        self._build_models()
        self._define_optimizers()

    def _build_models(self):
        self.image_encoder = CNNEncoder.load(self.cnn_weights, 256).to(self.device)
        self.text_encoder = RNNEncoder.load(self.rnn_weights, self.n_words).to(self.device)

        self.generator = Generator().to(self.device)
        self.discriminators = [
            Discriminator64().to(self.device),
            Discriminator128().to(self.device),
            Discriminator256().to(self.device)
        ]

    def _define_optimizers(self):
        betas = (0.5, 0.999)
        self.g_optim = Adam(self.generator.parameters(), lr=2e-4, betas=betas)
        self.d_optims = [Adam(d.parameters(), lr=2e-4, betas=betas) for d in self.discriminators]

    def _prepare_labels(self) -> Tuple[Tensor, Tensor, Tensor]:
        real_labels = torch.FloatTensor(self.batch_size).fill_(1).to(self.device)
        fake_labels = torch.FloatTensor(self.batch_size).fill_(0).to(self.device)
        match_labels = torch.LongTensor(range(self.batch_size)).to(self.device)

        return real_labels, fake_labels, match_labels

    def _save_model(self, epoch: int):
        os.makedirs(self.generator_path_save, exist_ok=True)
        os.makedirs(self.discriminators_path_save, exist_ok=True)

        path = os.path.join(self.generator_path_save, f"gen_weights_epoch_{epoch}.pth")
        torch.save(self.generator.state_dict(), path)

        for i, d in enumerate(self.discriminators):
            path = os.path.join(self.discriminators_path_save, f"d_{i}_weights_epoch_{epoch}.pth")
            torch.save(d.state_dict(), path)

    def _build_embeds(self, captions, captions_len) -> Tuple[Tensor, Tensor]:
        word_embeds, sentence_embeds = self.text_encoder(captions, captions_len)
        return word_embeds.detach(), sentence_embeds.detach()

    @staticmethod
    def _build_mask(word_embeds: Tensor, captions: Tensor) -> Tensor:
        mask = (captions == 0)
        num_words = word_embeds.shape[2]

        if mask.shape[1] > num_words:
            return mask[:, :num_words]

        return mask

    def train(self) -> Tuple[List[float], List[float]]:
        real_labels, fake_labels, match_labels = self._prepare_labels()

        g_losses_epoch, d_losses_epoch = [], []
        for epoch in trange(self.num_epochs, desc="Train AttnGAN"):

            g_losses, d_losses = [], []
            for batch in self.data_loader:
                print("Start")
                images, captions, captions_len, class_ids, _ = prepare_data(batch, self.device)
                word_embeds, sentence_embeds = self._build_embeds(captions, captions_len)
                mask = self._build_mask(word_embeds, captions)

                noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
                fake_images, _, mu, log_var = self.generator(noise, sentence_embeds, word_embeds, mask)

                d_total_loss = 0
                for i in range(len(self.discriminators)):
                    self.discriminators[i].zero_grad()
                    d_loss = self.loss.discriminator_loss(self.discriminators[i], images[i], fake_images[i],
                                                          sentence_embeds, real_labels, fake_labels)
                    d_loss.backward()
                    self.d_optims[i].step()
                    d_total_loss += d_loss

                self.generator.zero_grad()
                g_total_loss = self.loss.generator_loss(self.discriminators, self.image_encoder,
                                                        fake_images, real_labels, word_embeds, sentence_embeds,
                                                        match_labels, captions_len, class_ids)
                kl_loss = self.loss.kl_loss(mu, log_var)
                g_total_loss += kl_loss
                g_total_loss.backward()
                self.g_optim.step()

                g_losses.append(g_total_loss.item())
                d_losses.append(d_total_loss.item())

            g_losses_epoch.append(np.mean(g_losses))
            d_losses_epoch.append(np.mean(d_losses))

            print(f"\nEpoch: {epoch}")
            print(f"Generator average epoch loss: {round(g_losses_epoch[-1], 2)}")
            print(f"Discriminator average epoch loss: {round(d_losses_epoch[-1], 2)}\n")

            self._save_model(epoch)

        return g_losses_epoch, d_losses_epoch
