from WGAN.tools.config import opt
from WGAN.tools.model_define import Discriminator, Generator
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader


class WGAN:
    def __init__(self,
                 g: Generator,
                 d: Discriminator,
                 ):
        self.g = g
        self.d = d
        self.g_optimizer = torch.optim.RMSprop(self.g.parameters(), lr=opt.g_lr)
        self.d_optimizer = torch.optim.RMSprop(self.d.parameters(), lr=opt.d_lr)
        #  4: W GAN use RMSprop

    def __compute_g_loss(self,
                         fake_out: torch.Tensor) -> torch.Tensor:
        return -torch.mean(fake_out)

    def __compute_d_loss(self,
                         real_out: torch.Tensor,
                         fake_out: torch.Tensor) -> torch.Tensor:
        return -torch.mean(real_out) + torch.mean(fake_out)

    def __clamp_weight_d(self,
                         ):
        for p in self.d.parameters():
            p.data.clamp_(-opt.clamp_value, opt.clamp_value)

    def train_one_epoch(self,
                        data_loader: DataLoader,
                        ):
        self.g.train()
        self.d.train()

        device = next(self.g.parameters()).device

        for i, (real_img, _) in enumerate(data_loader):  # type: torch.Tensor

            real_img = real_img.to(device)
            batch_size = real_img.shape[0]

            if (i + 1) % opt.train_d_frequence == 0:

                random_noise = torch.randn(size=(batch_size, opt.g_noise_channel, 1, 1)).to(device)
                fake_img = self.g(random_noise).detach()  # care

                real_out = self.d(real_img)
                fake_out = self.d(fake_img)

                loss = self.__compute_d_loss(real_out, fake_out)
                #  2: W GAN compute loss

                self.d_optimizer.zero_grad()
                loss.backward()
                self.d_optimizer.step()

                self.__clamp_weight_d()
                #  3: W GAN clamp weight of d

            if (i + 1) % opt.train_g_frequence == 0:
                random_noise = torch.randn(size=(batch_size, opt.g_noise_channel, 1, 1)).to(device)
                fake_img = self.g(random_noise)
                fake_out = self.d(fake_img)

                loss = self.__compute_g_loss(fake_out)
                #  2: W GAN compute loss

                self.g_optimizer.zero_grad()
                self.d_optimizer.zero_grad()  # care
                loss.backward()
                self.g_optimizer.step()

    def train(self,
              data_loader: DataLoader,
              ):

        for epoch in tqdm(range(opt.MAX_EPOCH)):
            self.train_one_epoch(data_loader)
            self.eval(opt.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch))

    def eval(self,
             save_path: str = '',
             ):

        os.makedirs(save_path, exist_ok=True)

        self.g.eval()
        self.d.eval()
        device = next(self.g.parameters()).device

        random_noise = torch.randn(size=(opt.BATCH_SIZE, opt.g_noise_channel, 1, 1)).to(device)

        fake_img = self.g(random_noise)
        fake_img = fake_img * 0.5 + 0.5  # type:torch.Tensor

        fake_img = fake_img.cpu().detach().numpy()  # type: np.ndarray
        fake_img = np.transpose(fake_img, axes=(0, 2, 3, 1))

        for index in range(opt.BATCH_SIZE):
            plt.imsave('{}/{}.png'.format(save_path, index), fake_img[index])
