### Step 1: run SLR train
## change @config-file based on needed
python3 main.py --optimization savlr --admm-train --config-file config_resnet18_0.9 --batch-size 128

### Step 2: run maked retrain
## change @config-file based on needed
## change @admmtrain-acc based on model from admm train
python3 main.py --optimization savlr --masked-retrain --admmtrain-acc 76.08 --config-file config_resnet18_0.9 --batch-size 128 --ext _tmp3



### Step 1: run ADMM train
python3 main.py --optimization admm --admm-train --config-file config_resnet18_0.9 --batch-size 128

### Step 2: run maked retrain
python3 main.py --optimization admm --masked-retrain --admmtrain-acc 76.08 --config-file config_resnet18_0.9 --batch-size 128 --ext _tmp3
