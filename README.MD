```
gnnsyn/
├── config/
│   ├── mp_configs/
│   │   ├── GCN.json
│   │   ├── GraphSAGE.json
│   │   ├── GAT.json
│   │   └── ... (other operator configs)
│   └── task_configs/
│       ├── cora_node.json
│       ├── reddit_node.json
│       ├── pubmed_node.json
│       ├── ogbn_arxiv_node.json
│       └── ... (other dataset configs)
├── data/
│   └── dataset.py              # If you have custom dataset classes
├── model/
│   ├── pre_process.py          # PreProcessLayer
│   ├── message_passing.py      # MessagePassingLayer + forward_full / forward_subgraph
│   ├── temporal.py             # (Optional) RNNTemporal, etc.
│   ├── post_process.py         # PostProcessLayer
│   └── gnnsyn_model.py         # GNNsynModel (forward, forward_ns) orchestrating all modules
├── train/
│   └── trainer.py              # train_one_epoch_full, train_one_epoch_ns, eval_model_xxx, etc.
├── utils/
│   └── gnn_config_parser.py    # parse_gnn_config => merges mp_config + task_config, builds GNNsynModel
├── main.ipynb                  # Example notebook for running experiments (Cora, Reddit, etc.)
├── README.md                   # Project overview & instructions
└── requirements.txt            # (Optional) pinned package versions, PyTorch Geometric deps, etc.
```

依赖安装命令：
```
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

Install `gnnsyn` in your new environment (editable install)

A reliable method is to do an **editable install** so that `gnnsyn` is discovered automatically. For instance:

1. Go to your project’s **root directory**, the one containing the `gnnsyn/` folder.  
2. Create a `setup.py` or `pyproject.toml` (if you haven’t). For a minimal `setup.py`, you could use:

   ```python
   # setup.py
   from setuptools import setup, find_packages

   setup(
       name="gnnsyn",
       version="0.1.0",
       packages=find_packages(),
   )
   ```

3. Then run:
   ```bash
   pip install -e .
   ```
   - **`-e`** means *editable* mode: any changes in `gnnsyn/` are reflected immediately without reinstall.
