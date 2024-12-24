# gnnsyn/model/message_passing.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "swish":
        return nn.SiLU()
    else:
        return nn.Identity()

class MessagePassingLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 conv_type='gcn',
                 activation='relu',
                 dropout=0.0,
                 use_bn=False,
                 skip_type='none'):
        super().__init__()
        self.conv_type = conv_type
        self.skip_type = skip_type

        # Build conv
        if conv_type == 'gcn':
            self.conv = GCNConv(in_dim, out_dim, cached=False, normalize=True)
        elif conv_type == 'sage':
            self.conv = SAGEConv(in_dim, out_dim)
        elif conv_type == 'gat':
            self.conv = GATConv(in_dim, out_dim, heads=1)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        self.activation_fn = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        else:
            self.bn = nn.Identity()

        if skip_type != 'none':
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)

    ############### full-batch ###############
    def forward_full(self, x, edge_index):
        out = self.conv_forward_core(x, edge_index)
        out = self.apply_skip(x, out)
        out = self.bn_act_drop(out)
        return out

    ############### subgraph ###############
    def forward_subgraph(self, x_in, size, edge_index):
        """
        x_in: [N_sub, in_dim]
        size: [in_nodes, out_nodes]
        returns x_out: [out_nodes, out_dim]
        """
        # 1) conv
        if self.conv_type == 'sage':
            x_target = x_in[:size[1]]  # [out_nodes, in_dim]
            out = self.conv((x_in, x_target), edge_index)  # shape: [N_sub, out_dim]
        elif self.conv_type in ['gcn', 'gat']:
            out = self.conv(x_in, edge_index)
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")

        # 2) skip
        out = self.apply_skip(x_in, out, size=size)

        # 3) BN, act, drop
        out = self.bn_act_drop(out)

        # 4) 只保留 目标节点
        out_nodes = size[1]
        out = out[:out_nodes]  # => shape [out_nodes, out_dim]
        return out

    def conv_forward_core(self, x, edge_index):
        return self.conv(x, edge_index)

    def apply_skip(self, x_in, x_out, size=None):
        """
        skip-connection: 'none' / 'skip-sum' / 'skip-cat'
        we handle subgraph shape carefully if size is not None
        """
        if self.skip_type == 'none':
            return x_out
        # need projection if dims not match
        if x_in.size(-1) != x_out.size(-1):
            x_in_proj = self.skip_proj(x_in)
        else:
            x_in_proj = x_in

        if self.skip_type == 'skip-sum':
            # if subgraph => we want same # of rows
            if size is not None:
                out_nodes = size[1]
                x_in_proj = x_in_proj[:out_nodes]
                x_out = x_out[:out_nodes]  # we do final slice in forward_subgraph, but safe here
            return x_out + x_in_proj

        elif self.skip_type == 'skip-cat':
            if size is not None:
                out_nodes = size[1]
                x_in_proj = x_in_proj[:out_nodes]
                x_out = x_out[:out_nodes]
            return torch.cat([x_out, x_in_proj], dim=-1)
        else:
            raise ValueError(f"Unknown skip_type: {self.skip_type}")

    def bn_act_drop(self, x):
        if x.dim() == 2 and self.use_bn:
            x = self.bn(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x
