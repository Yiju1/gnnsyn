gnnsyn/
│
├── config/
│   └── gnn_arch_example.json     # 示例：存放 GNNsyn 需要的配置(JSON)
│
├── data/
│   └── dataset.py                # 数据加载 & 预处理模块
│
├── model/
│   ├── pre_process.py            # Pre-Process Layer
│   ├── message_passing.py        # Message Passing Layer
│   ├── temporal.py               # Temporal Layers (RNN/CNN/Attention)
│   ├── post_process.py           # Post-Process Layer
│   └── gnnsyn_model.py           # 整合各模块，形成完整 GNNsynModel
│
├── train/
│   └── trainer.py                # 训练 & 评测逻辑
│
├── utils/
│   └── config_parser.py          # 解析 JSON 配置文件，生成模型 & 训练参数
│
└── main.ipynb                    # Jupyter Notebook: Sanity Test (最常见的GNN & 数据集)
