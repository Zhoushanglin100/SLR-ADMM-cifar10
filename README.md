# SLR-cifar10-resnet18

## Instructions

1. Specify parameters in parameters.py file:

| Parameter             | Description
| --------------------- | --------------------------------------------- |
| `config_file`         | configuration for sparsity                    |
| `epochs`              | epochs for training                           |
| `optimization`        | 'savlr' for SLR and 'admm' for ADMM method    |
| `retraining_epochs`   | epochs for retraining(finetuning)             |
| `masked_retrain`      | flag for performing retraining                |
| `admm_train`          | flag for performing admm/slr training         |


2. Run with command `python3 main.py`

3. See `run.sh` for example

## Installation requirements

TODO! If you're having problems, try using PyTorch 1.7.1 and torchvision 0.8.2.

Uses at least these packages:

1. `torch`
2. `torchvision`
3. `numpy`
4. `yaml`
5. `pickle`


## Architecture

See `ARCHITECTURE.md` for more details

```
├── ckpts
│   ├── baseline
│   │   └── resnet18.pt
│   └── resnet18
│       └── savlr_train
│           ├── cifar10_resnet18_87.92_config_resnet18_0.9_pattern.pt
│           └── cifar10_resnet18_89.69_config_resnet18_0.9_pattern.pt
├── input_data.py
├── main.py
├── model
│   └── resnet.py
├── profile
│   ├── config_resnet18_0.9.yaml
│   ├── config_resnet18_v2.yaml
│   ├── config_resnet18_v4.yaml
│   ├── config_resnet18_v5.yaml
│   └── config_resnet18.yaml
├── README.md
├── README.txt
├── run.sh
├── slr
│   ├── admm_old.py
│   ├── admm.py
│   ├── parameters.py
│   └── testers.py
└── TODOs.md

```

TODO!
