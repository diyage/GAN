from Normal.tools.model_define import *

import torch
import os

a = torch.rand(size=(16, 3, 96, 96))
n = DiscriminatorNet()
y = n(a)
print(y)
m = GeneratorNet()
noise = torch.rand(size=(16, m.noise_channel, 1, 1))
out = m(noise)
print(out.shape)
b = y.to('cuda')
print(y.device)
print(b.device)
print(os.getcwd())
