from Normal.tools.model_define import GeneratorNet
import torch
import torch.nn as nn


class Generator:
    def __init__(self,
                 model: GeneratorNet,
                 optimizer: torch.optim.Optimizer = None):
        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), 2e-4, betas=(0.5, 0.999))
        else:
            self.optimizer = optimizer

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()