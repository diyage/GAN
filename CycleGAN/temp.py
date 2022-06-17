# from CycleGAN.tools.data_set_define import DoubleInputDataSet
# from torchvision.transforms import transforms
# from CycleGAN.tools.config import BaseConfig
# from torch.utils.data import DataLoader
#
# opt = BaseConfig()
#
# transforms_ = transforms.Compose([
#     transforms.Resize(int(opt.img_height*1.12)),
#     transforms.RandomCrop((opt.img_height, opt.img_width)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
# )
#
# data_set_train = DoubleInputDataSet(root=opt.data_set_root, transform=transforms_, training=True)
# data_loader_train = DataLoader(data_set_train, shuffle=True)
#
# data_set_test = DoubleInputDataSet(root=opt.data_set_root, transform=transforms_, training=False)
# data_loader_test = DataLoader(data_set_test, shuffle=True)
#
# for _, (a, b) in enumerate(data_loader_test):
#     print(a)
#     print(b)
#     print('*'*100)

import os
a = os.path.join('/a', '/b/', 'c')
print(a)
print(os.getcwd())
print(os.path.exists('E:\PyCharm//GAN/////CycleGAN'))