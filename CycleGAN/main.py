from CycleGAN.tools.model_define import Generator, Discriminator
from CycleGAN.tools.cycle_gan import CycleGAN
from CycleGAN.tools.config import BaseConfig
from CycleGAN.tools.data_set_define import DoubleInputDataSet

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


opt = BaseConfig()

transforms_ = transforms.Compose([
    transforms.Resize(int(opt.img_height*1.12)),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

data_set_train = DoubleInputDataSet(root=opt.data_set_root, transform=transforms_, training=True)
data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=opt.BATCH_SIZE)

data_set_test = DoubleInputDataSet(root=opt.data_set_root, transform=transforms_, training=False)
data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=opt.BATCH_SIZE)

g_a_to_b = Generator(opt.input_nc_A, opt.input_nc_B, opt.n_residual_blocks).to(opt.device)
g_b_to_a = Generator(opt.input_nc_B, opt.input_nc_A, opt.n_residual_blocks).to(opt.device)

d_a = Discriminator(opt.n_D_layers, opt.input_nc_A).to(opt.device)
d_b = Discriminator(opt.n_D_layers, opt.input_nc_B).to(opt.device)


w_gan = CycleGAN(g_a_to_b, g_b_to_a, d_a, d_b, opt)

w_gan.train(data_loader_train)

