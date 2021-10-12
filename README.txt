1. Specify parameters in parameters.py file.

config_file: configuration for sparsity
epochs: epochs for training
optimization: 'savlr' for SLR and 'admm' for ADMM method
retraining_epochs: epochs for retraining(finetuning)
masked_retrain: flag for performing retraining
admm_train: flag for performing admm/slr training


2. Run with command "python3 main.py"