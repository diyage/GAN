from Normal.tools.generator import Generator, GeneratorNet
from Normal.tools.discriminator import Discriminator, DiscriminatorNet
from Normal.tools.gan import NormalGAN
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

images = 'E:/tmp/'
trans_form = transforms.Compose([
    transforms.Resize(size=(96, 96)),
    transforms.CenterCrop(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
data_set = ImageFolder(images, transform=trans_form)
data_loader = DataLoader(data_set, shuffle=True)

g_net = GeneratorNet(noise_channel=128)
g = Generator(g_net)

d_net = DiscriminatorNet()
d = Discriminator(d_net)

gan = NormalGAN(g, d)

gan.train(data_loader, max_epoch=10000)


