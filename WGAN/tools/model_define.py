import torch
import torch.nn as nn
from WGAN.tools.config import opt


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        n_channel = opt.g_base_channel
        noise_channel = opt.g_noise_channel
        image_channel = opt.image_channel
        self.__net = nn.Sequential(
            nn.ConvTranspose2d(noise_channel, n_channel*8, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(n_channel*8),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel*8, n_channel * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 4, n_channel * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 2, n_channel * 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 1),
            nn.ReLU(),

            nn.ConvTranspose2d(n_channel * 1, image_channel, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return self.__net(x.view(-1, opt.g_noise_channel, 1, 1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        n_channel = opt.d_base_channel
        self.__net = nn.Sequential(
            nn.Conv2d(opt.image_channel, n_channel, kernel_size=(5, 5), stride=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(n_channel),
            nn.LeakyReLU(0.2),
            # 32*32*n_channel

            nn.Conv2d(n_channel, n_channel * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 2),
            nn.LeakyReLU(0.2),
            # 16*16*(n_channel*2)

            nn.Conv2d(n_channel * 2, n_channel * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 4),
            nn.LeakyReLU(0.2),
            # 8*8*(n_channel*4)

            nn.Conv2d(n_channel * 4, n_channel * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(n_channel * 8),
            nn.LeakyReLU(0.2),
            # 4*4*(n_channel*8)

            nn.Conv2d(n_channel * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            #  1: W GAN delete sigmoid of D
        )

    def forward(self, x: torch.Tensor):
        out = self.__net(x)  # type:torch.Tensor
        return out.view(-1)
