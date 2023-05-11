from transformers import AutoConfig
# from transformers import BertEncoder
import torch
from torch import nn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.Encoder = nn.Sequential(nn.Linear(1, hidden_dim//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim//2, hidden_dim))
        self.Decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim//2, 1))
    
    def forward(self, txt):
        #print('in VAE:', txt.dtype, txt.device)
        latent_rep = self.Encoder(txt)
        recons_txt = self.Decoder(latent_rep)
        return latent_rep, recons_txt

def vae_loss(txt, recons_txt, Encoder):
    loss = F.mse_loss(recons_txt, txt, reduction='mean')
    return loss
