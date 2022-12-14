from Normal.tools.generator import Generator, GeneratorNet
from Normal.tools.discriminator import Discriminator, DiscriminatorNet
from Normal.tools.gan import NormalGAN
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

images_folder = '/home/dell/data/Faces'
device = 'cuda:1'

trans_form = transforms.Compose([
    transforms.Resize(size=(96, 96)),
    transforms.CenterCrop(size=(96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
data_set = ImageFolder(images_folder, transform=trans_form)
data_loader = DataLoader(data_set, shuffle=True, batch_size=256)

g_net = GeneratorNet(noise_channel=100)
g = Generator(g_net)

d_net = DiscriminatorNet()
d = Discriminator(d_net)

gan = NormalGAN(g, d, device=device)

gan.train(data_loader, max_epoch=10000)


