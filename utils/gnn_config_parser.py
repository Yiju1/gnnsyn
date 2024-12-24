# gnnsyn/utils/gnn_config_parser.py

import json
import os
from gnnsyn.model.gnnsyn_model import GNNsynModel

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def merge_configs(mp_conf, task_conf):
    """
    将 mp_conf(算法相关) 和 task_conf(任务相关) 合并,
    返回一个 dict, 里面含有构建 GNNsyn 所需的所有字段.
    """
    merged = {}

    # 1) MP 层相关
    merged["num_mp_layers"] = mp_conf["num_mp_layers"]
    merged["mp_params"] = mp_conf["mp_params"]

    # 2) Task 相关: pre-process, post-process, model_io, learning_config
    merged["num_pre_layers"] = task_conf["num_pre_layers"]
    merged["pre_params"] = task_conf["pre_params"]
    merged["num_post_layers"] = task_conf["num_post_layers"]
    merged["post_params"] = task_conf["post_params"]

    merged["use_temporal"] = task_conf["use_temporal"]
    merged["temporal_params"] = task_conf["temporal_params"]

    merged["pool_type"] = task_conf["pool_type"]
    merged["task_type"] = task_conf["task_type"]

    merged["model_io"] = task_conf["model_io"]
    merged["learning_config"] = task_conf["learning_config"]

    # 3) Neighbor Sampling 开关 (若有)
    merged["use_ns"] = task_conf.get("use_ns", False)
    merged["neighbor_sizes"] = task_conf.get("neighbor_sizes", [25,10])

    return merged

def build_gnnsyn_model(merged_conf):
    """
    根据 merged_conf 中的字段构建 GNNsynModel 并返回。
    """
    io = merged_conf["model_io"]

    model = GNNsynModel(
        in_dim=io["in_dim"],
        hidden_dim=io["hidden_dim"],
        out_dim=io["out_dim"],
        num_pre_layers=merged_conf["num_pre_layers"],
        pre_params=merged_conf["pre_params"],
        num_mp_layers=merged_conf["num_mp_layers"],
        mp_params=merged_conf["mp_params"],
        num_post_layers=merged_conf["num_post_layers"],
        post_params=merged_conf["post_params"],
        use_temporal=merged_conf["use_temporal"],
        temporal_params=merged_conf["temporal_params"],
        pool_type=merged_conf["pool_type"],
        task_type=merged_conf["task_type"],
        use_ns=merged_conf["use_ns"]
    )
    return model

def parse_gnn_config(
    mp_config_file: str,
    task_config_file: str,
    single_model=True
):
    """
    读取并合并两个JSON文件, 并直接构建GNNsynModel。
    返回 (model, merged_conf).
    """
    mp_conf = load_json(mp_config_file)
    task_conf = load_json(task_config_file)

    # 1) 合并
    merged_conf = merge_configs(mp_conf, task_conf)

    # 2) 构建模型
    model = build_gnnsyn_model(merged_conf)

    # 3) 返回 (model, merged_conf)
    return model, merged_conf
