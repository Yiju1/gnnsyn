# gnnsyn/model/gnnsyn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .pre_process import PreProcessLayer
from .message_passing import MessagePassingLayer
from .temporal import RNNTemporal
from .post_process import PostProcessLayer

def get_pool_fn(pool_type):
    pool_type = pool_type.lower()
    if pool_type == 'mean':
        return global_mean_pool
    elif pool_type == 'max':
        return global_max_pool
    elif pool_type == 'sum':
        return global_add_pool
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")

class GNNsynModel(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_pre_layers=1,
                 pre_params=None,
                 num_mp_layers=2,
                 mp_params=None,
                 num_post_layers=1,
                 post_params=None,
                 use_temporal=False,
                 temporal_params=None,
                 pool_type='mean',
                 task_type='graph',
                 use_ns=False):
        super().__init__()
        self.task_type = task_type
        self.use_temporal = use_temporal
        self.use_ns = use_ns

        # PreProcess
        self.pre_layers = nn.ModuleList()
        cur_dim = in_dim
        if pre_params is None:
            pre_params = [{}]*num_pre_layers
        for i in range(num_pre_layers):
            cfg = pre_params[i] if i < len(pre_params) else {}
            out_size = cfg.get("out_dim", hidden_dim)
            layer = PreProcessLayer(
                in_dim=cur_dim,
                out_dim=out_size,
                activation=cfg.get("activation", "relu"),
                dropout=cfg.get("dropout", 0.0),
                use_norm=cfg.get("use_norm", False),
                func_type=cfg.get("func_type", "linear")
            )
            self.pre_layers.append(layer)
            cur_dim = out_size

        # Message Passing
        self.mp_layers = nn.ModuleList()
        if mp_params is None:
            mp_params = [{}]*num_mp_layers
        for i in range(num_mp_layers):
            cfg = mp_params[i] if i < len(mp_params) else {}
            out_size = cfg.get("out_dim", hidden_dim)
            mp_layer = MessagePassingLayer(
                in_dim=cur_dim,
                out_dim=out_size,
                conv_type=cfg.get("conv_type", "gcn"),
                activation=cfg.get("activation", "relu"),
                dropout=cfg.get("dropout", 0.0),
                use_bn=cfg.get("use_bn", False),
                skip_type=cfg.get("skip_type", "none")
            )
            self.mp_layers.append(mp_layer)
            # if skip-cat => dimension changes
            if cfg.get("skip_type", "none") == "skip-cat":
                cur_dim = cur_dim + out_size
            else:
                cur_dim = out_size

        # Temporal
        if use_temporal:
            if temporal_params is None:
                temporal_params = {}
            rnn_type = temporal_params.get("rnn_type", "rnn")
            self.temporal_module = RNNTemporal(cur_dim, hidden_dim, rnn_type=rnn_type)
            cur_dim = hidden_dim

        # PostProcess
        self.post_layers = nn.ModuleList()
        if post_params is None:
            post_params = [{}]*num_post_layers
        for i in range(num_post_layers):
            cfg = post_params[i] if i < len(post_params) else {}
            out_size = cfg.get("out_dim", hidden_dim)
            layer = PostProcessLayer(
                in_dim=cur_dim,
                out_dim=out_size,
                activation=cfg.get("activation", "relu"),
                dropout=cfg.get("dropout", 0.0),
                use_norm=cfg.get("use_norm", False)
            )
            self.post_layers.append(layer)
            cur_dim = out_size

        # Pooling
        if task_type == 'graph':
            self.pool_fn = get_pool_fn(pool_type)
        else:
            self.pool_fn = None

        # final out
        self.fc_out = nn.Linear(cur_dim, out_dim)

    def forward(self, data):
        if self.use_ns:
            raise ValueError("use_ns=True, call forward_ns(...)!")
        x, edge_index = data.x, data.edge_index

        # pre
        for layer in self.pre_layers:
            x = layer(x)

        # mp
        for mp_layer in self.mp_layers:
            x = mp_layer.forward_full(x, edge_index)

        # temporal
        if self.use_temporal:
            pass

        # post
        for pp_layer in self.post_layers:
            x = pp_layer(x)

        # pool
        if self.pool_fn is not None and hasattr(data, 'batch'):
            x = self.pool_fn(x, data.batch)

        out = self.fc_out(x)
        return out

    def forward_ns(self, x, adjs):
        if not self.use_ns:
            raise ValueError("use_ns=False, call forward(...)!")
        # x: [N_sub, in_dim]
        # adjs: list of (edge_index, e_id, size)

        # pre
        for layer in self.pre_layers:
            x = layer(x)

        # mp
        for i, mp_layer in enumerate(self.mp_layers):
            (edge_index, _, size) = adjs[i]
            x = mp_layer.forward_subgraph(x, size, edge_index)
            # x => [out_nodes, out_dim]

        # temporal
        if self.use_temporal:
            pass

        # post
        for pp_layer in self.post_layers:
            x = pp_layer(x)

        # no pool if node-level
        out = self.fc_out(x)
        return out
