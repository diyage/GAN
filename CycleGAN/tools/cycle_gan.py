import torch
import torch.nn as nn
from CycleGAN.tools.model_define import Generator, Discriminator
from CycleGAN.tools.config import BaseConfig
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class CycleGAN:
    def __init__(self,
                 g_a_to_b: Generator,
                 g_b_to_a: Generator,
                 d_a: Discriminator,
                 d_b: Discriminator,
                 opt: BaseConfig,):
        self.opt = opt
        self.g_a_to_b = g_a_to_b
        self.g_b_to_a = g_b_to_a
        self.d_a = d_a
        self.d_b = d_b

        self.optimizer_g = torch.optim.Adam([*self.g_a_to_b.parameters(),
                                             *self.g_b_to_a.parameters()],
                                            lr=self.opt.lr,
                                            betas=(self.opt.b1, self.opt.b2),)
        self.optimizer_d_a = torch.optim.Adam(
            self.d_a.parameters(),
            lr=self.opt.lr,
            betas=(self.opt.b1, self.opt.b2)
        )
        self.optimizer_d_b = torch.optim.Adam(
            self.d_b.parameters(),
            lr=self.opt.lr,
            betas=(self.opt.b1, self.opt.b2)
        )

        self.loss_func_gan = nn.MSELoss()
        self.loss_func_cycle = nn.L1Loss()
        self.loss_func_identity = nn.L1Loss()

        self.patch = (1,
                      self.opt.img_height//(2**self.opt.n_D_layers) - 2,
                      self.opt.img_width//(2**self.opt.n_D_layers) - 2)

    def __zero_grad(self):
        self.optimizer_d_a.zero_grad()
        self.optimizer_d_b.zero_grad()

        self.optimizer_g.zero_grad()

    def __train_g(self,
                  real_a: torch.Tensor,
                  real_b: torch.Tensor,
                  ):

        self.g_a_to_b.train()
        self.g_b_to_a.train()

        # Identity loss
        iden_a = self.g_b_to_a(real_a)
        iden_b = self.g_a_to_b(real_b)

        loss_identity = (self.loss_func_identity(iden_a, real_a) + self.loss_func_identity(iden_b, real_b)) / 2

        # GAN loss
        targets_for_g = torch.ones(size=(real_a.shape[0], *self.patch)).to(self.opt.device)

        fake_a = self.g_b_to_a(real_b)
        pred_fake_a = self.d_a(fake_a)

        fake_b = self.g_a_to_b(real_a)
        pred_fake_b = self.d_b(fake_b)

        loss_gan = (self.loss_func_gan(pred_fake_a, targets_for_g) + self.loss_func_gan(pred_fake_b, targets_for_g))/2

        # Cycle loss

        recov_a = self.g_b_to_a(fake_b)
        recov_b = self.g_a_to_b(fake_a)
        loss_cycle = (self.loss_func_cycle(recov_a, real_a) + self.loss_func_cycle(recov_b, real_b))/2

        # total loss
        loss = loss_gan + self.opt.lambda_cyc * loss_cycle + self.opt.lambda_id*loss_identity  # type:torch.Tensor

        self.__zero_grad()
        loss.backward()
        self.optimizer_g.step()

    def __train_d(self,
                  real_a: torch.Tensor,
                  real_b: torch.Tensor,
                  ):
        self.d_a.train()
        self.d_b.train()
        # train d_a
        targets_for_d_real = torch.ones(size=(real_a.shape[0], *self.patch)).to(self.opt.device)
        targets_for_d_fake = torch.zeros(size=(real_a.shape[0], *self.patch)).to(self.opt.device)

        fake_a = self.g_b_to_a(real_b)
        pred_fake_a = self.d_a(fake_a)
        pred_real_a = self.d_a(real_a)
        loss_d_a = 0.5 * (self.loss_func_gan(pred_fake_a,
                                             targets_for_d_fake) + self.loss_func_gan(pred_real_a,
                                                                                      targets_for_d_real))
        self.__zero_grad()
        loss_d_a.backward()
        self.optimizer_d_a.step()

        # train d_b
        fake_b = self.g_a_to_b(real_a)
        pre_fake_b = self.d_b(fake_b)
        pred_real_b = self.d_b(real_b)
        loss_d_b = 0.5 * (self.loss_func_gan(pre_fake_b,
                                             targets_for_d_fake) + self.loss_func_gan(pred_real_b,
                                                                                      targets_for_d_real))
        self.__zero_grad()
        loss_d_b.backward()
        self.optimizer_d_b.step()

    def __train_one_epoch(self,
                          data_loader: DataLoader,
                          ):
        for i, (real_a, real_b) in enumerate(data_loader):
            real_a = real_a.to(self.opt.device)
            real_b = real_b.to(self.opt.device)

            if (i+1) % self.opt.train_d_fre == 0:
                self.__train_d(real_a, real_b)

            if (i+1) % self.opt.train_g_fre == 0:
                self.__train_g(real_a, real_b)

    def eval(self,
             save_path: str,
             data_loader: DataLoader,
             ):
        def eval_one_generator(g: Generator, real_img, save_to: str):
            g.eval()
            fake_img = g(real_img)
            fake_img = fake_img * 0.5 + 0.5  # type:torch.Tensor

            fake_img = fake_img.cpu().detach().numpy()  # type: np.ndarray
            fake_img = np.transpose(fake_img, axes=(0, 2, 3, 1))

            for index in range(self.opt.BATCH_SIZE):
                os.makedirs(save_to, exist_ok=True)
                saved_file_name = os.path.join(save_to, '{}.png'.format(index))
                plt.imsave(saved_file_name, fake_img[index])

        for _, (real_a, real_b) in enumerate(data_loader):
            save_path_for_a = os.path.join(save_path, 'fake_a_style')
            save_path_for_b = os.path.join(save_path, 'fake_b_style')
            eval_one_generator(self.g_a_to_b, real_a.to(self.opt.device), save_to=save_path_for_b)
            eval_one_generator(self.g_b_to_a, real_b.to(self.opt.device), save_to=save_path_for_a)

    def train(self,
              data_loader_train: DataLoader,
              data_loader_test: DataLoader,
              ):
        for epoch in tqdm(range(self.opt.MAX_EPOCH)):
            self.__train_one_epoch(data_loader_train)
            save_path = '{}/{}/{}/{}'.format(self.opt.ABS_PATH, os.getcwd(), 'eval_images', epoch)
            self.eval(save_path, data_loader_test)





