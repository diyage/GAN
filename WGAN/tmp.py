from WGAN.tools.config import opt
from WGAN.tools.model_define import Discriminator
print(opt.train_d_frequence)
a = Discriminator()
print(next(a.parameters()).device)