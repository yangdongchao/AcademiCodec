import torch
import torch.nn as nn
import json

from models import Generator, Quantizer, Encoder
from env import AttrDict


class VQVAE(nn.Module):
    def __init__(self, config_path, ckpt_path, with_encoder=False):
        super(VQVAE, self).__init__()
        ckpt = torch.load(ckpt_path)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt['encoder'])

    def forward(self, x):
        # x is the codebook
        # print('x ', x.shape)
        # assert 1==2
        return self.generator(self.quantizer.embed(x)) # 

    def encode(self, x):
        batch_size = x.size(0)
        c = self.encoder(x.unsqueeze(1))
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        # print(torch.stack(c,-1).shape)
        # assert 1==2
        return torch.stack(c, -1) #N, T, 4
