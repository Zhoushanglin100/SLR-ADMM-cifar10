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
|   │ TODO - Description
│   ├── baseline
│   │     TODO - Description  
│   └── resnet18
│       │ TODO - Description
│       └── savlr_train
│             TODO - Description
│
├── input_data.py
│     TODO - Description
│ 
├── main.py
│     TODO - Description
│
├── model
│     TODO - Description
├── profile
│     TODO - Description
│
├── run.sh
│     TODO - Description
│
├── slr
│   ├── admm.py
│   │     TODO - Description
│   │
│   └── testers.py
│         TODO - Description
│
└── TODOs.md
```

TODO!
