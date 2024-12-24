# gnnsyn/model/pre_process.py

import torch.nn as nn

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "swish":
        return nn.SiLU()  # PyTorch中SiLU等价于Swish
    else:
        return nn.Identity()

class PreProcessLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation='relu',
                 dropout=0.0,
                 use_norm=False,
                 func_type='linear'):
        super().__init__()
        self.use_norm = use_norm
        if use_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = nn.Identity()

        self.activation_fn = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

        if func_type == 'linear':
            self.fc = nn.Linear(in_dim, out_dim)
        elif func_type == 'mlp':
            self.fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
        else:
            raise ValueError("Unknown func_type for PreProcessLayer")

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x
