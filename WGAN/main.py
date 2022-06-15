from WGAN.tools.model_define import Generator, Discriminator
from WGAN.tools.wgan import WGAN
from WGAN.tools.config import opt

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


trans_form = transforms.Compose([
    transforms.Resize(size=opt.image_size),
    transforms.CenterCrop(size=opt.image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
data_set = ImageFolder(opt.images_folder, transform=trans_form)
data_loader = DataLoader(data_set, shuffle=True, batch_size=opt.BATCH_SIZE)

g = Generator()
g.to(opt.device)
d = Discriminator()
d.to(opt.device)

w_gan = WGAN(g, d)

w_gan.train(data_loader)

