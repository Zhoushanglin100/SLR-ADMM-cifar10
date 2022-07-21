
seed = 1
batch = 128
test_batch = 1000
rho = 0.1

rho_num = 1 #define how many rohs for ADMM training
config_file = 'config_resnet18_v3' #prune config file
sparsity_type = 'irregular' #irregular,column,filter,pattern,random-pattern
mylr = 0.01 #learning rate
momentum = 0.5 #SGD momentum (default: 0.5)
epochs = 50#number of epochs to train (default: 2)
retrain_epoch = 100# #for retraining
lr_scheduler = 'cosine'
combine_progressive = False #for filter pruning after column pruning

verbose = False #whether to report admm convergence condition
lr_decay = 30 #how many every epoch before lr drop (default: 30)
optmzr = 'sgd' #optimizer used (default: sgd)
log_interval = 100 #how many batches to wait before logging training status
save_model = "pretrained_mnist.pt" #For Saving the current Model
load_model = None #For loading the model

optimization = 'savlr' #'admm' or 'savlr'

masked_retrain = True #for masked training
admm_train =  True #for admm training    

admm_epoch = 10 #how often we do admm update
    
#SAVLR parameters:
M = 300
r = 0.1
initial_s = 0.01
     