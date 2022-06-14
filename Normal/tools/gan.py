from Normal.tools.generator import Generator
from Normal.tools.discriminator import Discriminator
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

ABS_PATH = ''


class NormalGAN:
    def __init__(self,
                 g: Generator,
                 d: Discriminator,
                 train_d_frequence: int = 1,
                 train_g_frequence: int = 5,
                 device: str = 'cpu'):
        self.g = g
        self.d = d

        self.device = device
        self.g.model.to(self.device)
        self.d.model.to(self.device)

        self.train_d_fre = train_d_frequence
        self.train_g_fre = train_g_frequence
        self.bce_loss_func = nn.BCELoss()

    def train_one_epoch(self,
                        data_loader: DataLoader,
                        ):
        self.g.train()
        self.d.train()

        for i, (real_img, _) in enumerate(data_loader):  # type: torch.Tensor
            real_img = real_img.to(self.device)
            batch_size = real_img.shape[0]

            if (i + 1) % self.train_d_fre == 0:

                random_noise = torch.randn(size=(batch_size, self.g.model.noise_channel, 1, 1)).to(self.device)
                fake_img = self.g.model(random_noise).detach()  # care

                real_out = self.d.model(real_img)
                fake_out = self.d.model(fake_img)

                real_target_for_d = torch.ones(size=real_out.shape).to(self.device)
                fake_target_for_d = torch.zeros(size=fake_out.shape).to(self.device)

                loss = self.bce_loss_func(real_out, real_target_for_d) + self.bce_loss_func(fake_out, fake_target_for_d)
                self.d.optimizer.zero_grad()
                loss.backward()
                self.d.optimizer.step()

            if (i + 1) % self.train_g_fre == 0:
                random_noise = torch.randn(size=(batch_size, self.g.model.noise_channel, 1, 1)).to(self.device)
                fake_img = self.g.model(random_noise)
                fake_out = self.d.model(fake_img)
                fake_target_for_g = torch.ones(size=fake_out.shape).to(self.device)
                loss = self.bce_loss_func(fake_out, fake_target_for_g)
                self.g.optimizer.zero_grad()
                self.d.optimizer.zero_grad()  # care
                loss.backward()
                self.g.optimizer.step()

    def train(self,
              data_loader: DataLoader,
              max_epoch):

        for _ in tqdm(range(max_epoch)):
            self.train_one_epoch(data_loader)
            self.eval(ABS_PATH + os.getcwd() + '/eval_images/')

    def eval(self,
             save_path: str = '',
             batch_size: int = 128,):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.g.eval()
        self.d.eval()

        random_noise = torch.randn(size=(batch_size, self.g.model.noise_channel, 1, 1)).to(self.device)
        fake_img = self.g.model(random_noise)
        fake_img = fake_img * 0.5 + 0.5  # type:torch.Tensor

        fake_img = fake_img.cpu().detach().numpy()  # type: np.ndarray
        fake_img = np.transpose(fake_img, axes=(0, 2, 3, 1))

        for index in range(batch_size):
            plt.imsave('{}/{}.png'.format(save_path, index), fake_img[index])













