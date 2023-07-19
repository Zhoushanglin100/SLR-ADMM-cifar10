export optim=savlr     # choose from [savlr, admm]
export model=18
export ratio=0.7
export bz=128
export ext=_tmp1

### Step 1: run SLR / ADMM train
## change @config-file based on needed
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
            --optimization $optim \
            --arch resnet$model \
            --admm-train \
            --config-file config_resnet$model\_$ratio \
            --baseline-model ckpts/baseline/resnet$model.pt \
            --batch-size $bz \
            --ext $ext

### Step 2: run masked retrain
## change @config-file based on needed
CUDA_VISIBLE_DEVICES=$1 python3 main.py \
            --optimization $optim \
            --arch resnet$model \
            --masked-retrain \
            --config-file config_resnet$model\_$ratio \
            --batch-size $bz \
            --ext $ext

