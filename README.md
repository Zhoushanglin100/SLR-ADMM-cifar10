# SLR-cifar10-resnet18

This is an example for train ADMM or SLR on cifar10 under Resnet18.

## Parameters.

config_file: configuration for sparsity on each layer
optimization: 'savlr' for SLR and 'admm' for ADMM method
retraining_epochs: epochs for retraining(finetuning)
masked_retrain: flag for performing retraining
admm_train: flag for performing admm/slr training


## Step 1: Admm / SLR Training
```bash
python3 main.py --optimization <optimization method> --admm-train --config-file <config file>

# e.g.: choose SLR optimization method, train model with each layer has compression rate being 0.9
python3 main.py --optimization savlr --admm-train --config-file config_resnet18_0.9
```

- Some specific parameters for SLR:
    - M: SLR stepsize-setting parameter M
    - r: SLR stepsize-setting parameter r
    - initial-s: SLR parameter initial stepsize
    - rho: define rho for ADMM")
    - rho-num: define how many rohs for ADMM training

- Parameter for SLR and ADMM both
    - config-file: help="prune config file")
    - sparsity-type: choose from [irregular, column, channel, filter, pattern, random-pattern]


## Step 2: run maked retrain
```bash
python3 main.py --optimization <optimization method> --masked-retrain --admmtrain-acc <previous admm/slr trained best acc> --config-file <config file>

# E.g.: 
python3 main.py --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_resnet18_0.9 
```

- Parameter for SLR and ADMM both
    - admmtrain-acc: SLR/ADMM trained best acc for saved model')


## Run detail: check run.sh