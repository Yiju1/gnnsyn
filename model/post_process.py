# gnnsyn/model/post_process.py

import torch.nn as nn
from .pre_process import get_activation

class PostProcessLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation='relu', dropout=0.0, use_norm=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = nn.Identity()

        self.activation_fn = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x
