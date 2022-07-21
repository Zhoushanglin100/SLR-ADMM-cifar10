# SLR-cifar10

This is an example for train ADMM or SLR on cifar10

-----------------------------------------------------------------------------

## Pruning
```bash
cd slr_pruning
```

### Parameters.

config_file: configuration for sparsity on each layer
optimization: 'savlr' for SLR and 'admm' for ADMM method
retraining_epochs: epochs for retraining(finetuning)
masked_retrain: flag for performing retraining
admm_train: flag for performing admm/slr training


### Step 1: Admm / SLR Training
```bash
python3 main.py --optimization <optimization method> --admm-train --config-file <config file>

# e.g.: choose SLR optimization method, train model with each layer has compression rate being 0.9
python3 main.py --optimization savlr --admm-train --config-file config_resnet18_0.9
```

- Some specific parameters for SLR:
    - M: SLR stepsize-setting parameter M
    - r: SLR stepsize-setting parameter r
    - initial-s: SLR parameter initial stepsize
    - rho: define rho
    - rho-num: define how many rohs

- Parameter for SLR and ADMM both
    - config-file: prune config file, define pruning layer and compression rate for each layer
    - sparsity-type: choose from [irregular, column, channel, filter, pattern, random-pattern]


### Step 2: run maked retrain
```bash
python3 main.py --optimization <optimization method> --masked-retrain --admmtrain-acc <previous admm/slr trained best acc> --config-file <config file>

# E.g.: 
python3 main.py --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_resnet18_0.9 
```

- Parameter for SLR and ADMM both
    - admmtrain-acc: SLR/ADMM trained best acc for saved model')


**Run detail: check run.sh**


-----------------------------------------------------------------------------

## Quantization
```bash
cd slr_quantization
```

There is only one step for quantization

```bash
python3 main_cifar10_new.py --optimization <optimization method> --admm-quant --load-model-name <pretrained model weight that want to quantize> -a <model architecture> --quant-type <quantization method>

# E.g.:
python3 main_cifar10_new.py --optimization admm --admm-quant --load-model-name "base/baseline_vgg16.pt" -a vgg16 --quant-type ternary

```

- Some specific parameters for SLR:
    - M: SLR stepsize-setting parameter M
    - r: SLR stepsize-setting parameter r
    - initial-s: SLR parameter initial stepsize

- Parameter for SLR and ADMM both
    - quant-type: : define sparsity type, choose from  [binary, ternary, fixed, one-size]
    - num-bits: If use fixed and and one-size number bits, please set bit length
    - update-rho: Choose whether to update initial rho in each iteration, 1-update, 0-not update
    - init-rho: initial rho for all layers

**Run detail: check run.sh**
