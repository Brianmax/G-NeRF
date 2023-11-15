import torch
import torch.nn as nn
import random

from model.rendering import Embedding
from diffAugment import DiffAugment

class Discriminator(nn.Module):
    def __init__(self, conditional, policy, ndf=64, imsize=64):
        super(Discriminator, self).__init__()
        nc = 3
        self.conditional = conditional
        self.policy = policy
        self.imsize = imsize

        # Simplificado: se eliminan algunas verificaciones y se simplifica la creación de capas
        layers = [
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1 if not conditional else ndf, 4, 1, 0),
            nn.LeakyReLU(0.2)
        ]

        self.main = nn.Sequential(*layers)

        if conditional:
            self.embedding_scale = Embedding(1, 4)
            self.final = nn.Sequential(
                nn.Conv2d(ndf + self.embedding_scale.out_channels, 1, 1),
                nn.LeakyReLU(0.2)
            )

    def forward(self, input, y=None):
        if self.policy is not None and random.random() > 0.5:
            input = DiffAugment(input, policy=self.policy)
        output = self.main(input)
        if self.conditional:
            y = self.embedding_scale(y, dim=1)
            output = torch.cat((output, y), 1)
            output = self.final(output).flatten()
        return output
