from __future__ import print_function
import argparse
import os, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import slr.admm as admm
# from slr.parameters import *
from model.resnet import resnet18

from input_data import CIFAR10DataLoader as DataLoader

try:
    import wandb
    has_wandb = True
    wandb.init(project='xxx', entity='xxxxxxx')
except ImportError: 
    has_wandb = False

import logging
LOG_FILENAME = 'output.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

writer = None
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'

torch.cuda.empty_cache()

########################################################################################

### Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--max-step', type=int, default=6000, metavar='N',
                    help='number of max step to train (default: 6000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-scheduler', type=str, default='cosine',
                    help="[default, cosine]")
parser.add_argument('--lr-decay', type=int, default=30, metavar='LR',
                    help='how many every epoch before lr drop (default: 30)')                    
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M',
                    help='Optimizer weight decay (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-dir', type=str, default="./ckpts", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--baseline-model', type=str, default="resnet18.pt", metavar='N',
                    help='For loading the model')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='whether to report admm convergence condition')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='Just run inference and evaluate on test dataset')
parser.add_argument('--optimizer-type', type=str, default='sgd',
                    help="choose optimizer type: [sgd, adam]")
parser.add_argument('--logger', action='store_true', default=False,
                    help='whether to use logger')

# -------------------- SLR Parameter ---------------------------------

parser.add_argument('--admm-train', action='store_true', default=False,
                    help='Choose admm quantization training')
parser.add_argument('--masked-retrain', action='store_true', default=False,
                    help='whether to masked training for admm quantization')
parser.add_argument('--optimization', type=str, default='savlr',
                    help='optimization type: [savlr, admm]')
parser.add_argument('--admm-epochs', type=int, default=10, metavar='N',
                    help='number of interval epochs to update admm (default: 1)')
parser.add_argument('--retrain-epoch', type=int, default=50, metavar='N',
                    help='for retraining')
parser.add_argument('--combine-progressive', action='store_true', default=False,
                    help='for filter pruning after column pruning')

parser.add_argument('--M', type=int, default=300, metavar='N',
                    help='SLR parameter M ')
parser.add_argument('--r', type=float, default=0.1, metavar='N',
                    help='SLR parameter r ')
parser.add_argument('--initial-s', type=float, default=0.01, metavar='N',
                    help='SLR parameter initial stepsize')
parser.add_argument('--rho', type=float, default=0.1, 
                    help="define rho for ADMM")
parser.add_argument('--rho-num', type=int, default=1, 
                    help="define how many rohs for ADMM training")

parser.add_argument('--config-file', type=str, default='config_resnet18_0.9', 
                    help="prune config file")
parser.add_argument('--sparsity-type', type=str, default='pattern',
                    help='sparsity type: [irregular,column,channel,filter,pattern,random-pattern]')
parser.add_argument('--admmtrain-acc', type=float, default=76.08, metavar='N',
                    help='SLR trained best acc for saved model')
parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')

parser.add_argument('--ext', type=str, default="", metavar='N',
                    help='extension of file name')

####################################################################################

args, unknown = parser.parse_known_args()
if has_wandb:
    wandb.init(config=args)
    wandb.config.update(args)

####################################################################################


def train(args, ADMM, model, device, train_loader, optimizer, epoch, writer):
   
    model.train()

    ce_loss = None
    mixed_loss = None
    ctr=0
    total_ce = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        ctr += 1
        mixed_loss_sum = []
        loss = []

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        ce_loss = F.cross_entropy(output, target)
        total_ce = total_ce + float(ce_loss.item())
        
        admm.z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        
        ce_loss, admm_loss, mixed_loss = admm.append_admm_loss(args, ADMM, model, ce_loss)  # append admm losss

        #mixed_loss.backward()
        mixed_loss.backward(retain_graph=True)
        optimizer.step()
 
        mixed_loss_sum.append(float(mixed_loss))
        loss.append(float(ce_loss))

        if batch_idx % args.print_freq == 0:
            print("Cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))
           
    lossadmm = []
    for k, v in admm_loss.items():
            # print("at layer {}, admm loss is {}".format(k, v))
            lossadmm.append(float(v))

    #for k in ADMM.prune_ratios:
    #    writer.add_scalar('layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], epoch)
    
    if args.verbose:
        writer.add_scalar('Train/Cross_Entropy', ce_loss, epoch)
        for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))
            ADMM.admmloss[k].extend([float(v)])

        for k in ADMM.prune_ratios:
            writer.add_scalar('layer:{} Train/ADMM_Loss'.format(k), admm_loss[k], epoch)
            
    ADMM.ce_prev = ADMM.ce
    ADMM.ce = total_ce / ctr
    
    return mixed_loss_sum, loss


def test(args, model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset))


def total_params(model):
    return sum([np.prod(param.size()) for param in model.parameters()])

def param_to_array(param):
    return param.cpu().data.numpy().reshape(-1)

def get_sorted_list_of_params(model):
    params = list(model.parameters())
    param_arrays = [param_to_array(param) for param in params]
    return np.sort(np.concatenate(param_arrays))

####################################################################################
#MAIN STARTS HERE:

def main():

    logging.info('Optimization: ' + args.optimization)
    logging.info('Epochs: ' + str(args.epochs))
    logging.info('rho: ' + str(args.rho))
    logging.info('Retraining Epochs: ' + str(args.retrain_epoch))

    logging.info('SLR stepsize: ' + str(args.initial_s))

    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = DataLoader(args.batch_size, args.test_batch_size, kwargs)
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader

    if args.arch == "resnet18":
        model = resnet18().to(device)
    print("\nArch name is {}".format(args.arch))
    
    if has_wandb:
        wandb.watch(model)

    # for i, (name, W) in enumerate(model.named_parameters()):
    #     print(i, "th weight:", name, ", shape = ", W.shape, ", weight.dtype = ", W.dtype)  

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("The optimizer type is not defined!")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=4e-08)


    """=============="""
    """  ADMM Train  """
    """=============="""

    initial_rho = args.rho
    if args.admm_train:

        if has_wandb == False:
            condition_d = {}
            mixed_losses = []
            ce_loss = []
            test_acc = []

        for i in range(args.rho_num):
            current_rho = initial_rho * 10 ** i

            if i == 0:
                model_path = os.path.join(args.save_dir, "baseline", args.baseline_model)
                print("Path is:{}".format(model_path))

                model.load_state_dict(torch.load(model_path))  # admm train need basline model
                model.cuda()

                ac = test(args, model, device, test_loader)
                print("Initial model accuracy:")
                print(ac)
                logging.info('Initial model accuracy: ' + str(ac))
            else:
                model.load_state_dict(torch.load("model_prunned/mnist_{}_{}_{}.pt".format(current_rho/10, args.config_file, args.sparsity_type)))
                model.cuda()

            ADMM = admm.ADMM(args, model, "profile/" + args.config_file + ".yaml", rho=current_rho)
            admm.admm_initialization(args, ADMM, model)  # intialize Z and U variables


            # ----------------------------

            best_prec1 = 0.

            for epoch in range(1, args.epochs + 1):

                print("Epoch ", epoch)

                if args.lr_scheduler == 'default':
                    admm.admm_adjust_learning_rate(args, optimizer, epoch)
                elif args.lr_scheduler == 'cosine':
                    scheduler.step()
                print("current rho: {}".format(current_rho))
                
                if args.combine_progressive:
                    admm_loss, mixed_loss, loss = admm.admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch)
                else:
                    mixed_loss, loss = train(args, ADMM, model, device, train_loader, optimizer, epoch, writer)
                
                prec1 = test(args, model, device, test_loader)

                if has_wandb:
                    wandb.log({"test_acc": prec1})
                    wandb.log({"ce_loss": loss[0]})
                    wandb.log({"mixed_losses": mixed_loss[0]})
                else:
                    ce_loss.append(loss)
                    mixed_losses.append(mixed_loss)
                    test_acc.append(prec1)

                ### save temporary best acc model
                if (best_prec1 < prec1) and (epoch != 1):

                    ## remove old model
                    old_file = "cifar10_{}_{}_{}_{}{}.pt".format(args.arch, 
                                                                 best_prec1, 
                                                                 args.config_file, 
                                                                 args.sparsity_type, 
                                                                 args.ext)
                    if os.path.exists(args.save_dir+"/"+args.arch+"/"+args.optimization+"_train/"+old_file):
                        os.remove(args.save_dir+"/"+args.arch+"/"+args.optimization+"_train/"+old_file)
                    
                    ### save new one
                    best_prec1 = max(prec1, best_prec1)
                    model_best = model
                    torch.save(model_best.state_dict(), 
                                args.save_dir+"/{}/{}_train/cifar10_{}_{}_{}_{}{}.pt".format(args.arch,
                                                                                            args.optimization, 
                                                                                            args.arch,
                                                                                            best_prec1, 
                                                                                            args.config_file, 
                                                                                            args.sparsity_type,
                                                                                            args.ext))
                    if has_wandb:
                        wandb.log({"best_prec1": best_prec1})
                
                if args.optimization == "savlr":
                    print("Condition 1")
                    print(ADMM.condition1)
                    print("Condition 2")
                    print(ADMM.condition2)
                    
                    if has_wandb == False:
                        condition_d["Condition1"] = condition_d.get("Condition1", [])+ADMM.condition1
                        condition_d["Condition2"] = condition_d.get("Condition2", [])+ADMM.condition2

            print("----------------> Accuracy after hard-pruning ...")
            model_forhard = model_best
            admm.hard_prune(args, ADMM, model_forhard)
            admm.test_sparsity(args, ADMM, model_forhard)
            prec_hp = test(args, model, device, test_loader)

            hp_dir = args.save_dir+"/"+args.optimization+"_hp/"
            if not os.path.exists(hp_dir):
                os.makedirs(hp_dir)
            torch.save(model_forhard.state_dict(), 
                       hp_dir+"cifar10_{}_{}_{}_{}{}.pt".format(args.arch, 
                                                                prec_hp, 
                                                                args.config_file, 
                                                                args.sparsity_type,
                                                                args.ext))

            ### save result
            if has_wandb == False:

                if not os.path.exists(args.save_dir+"/results"):
                    os.makedirs(args.save_dir+"/results")

                f = open(args.save_dir+"/results/test_acc.pkl", "wb")
                pickle.dump(test_acc, f)
                f.close()

                f = open(args.save_dir+"/results/mixed_losses{}.pkl".format(current_rho),"wb")
                pickle.dump(mixed_losses,f)
                f.close()

                f = open(args.save_dir+"/results/ce_loss{}.pkl".format(current_rho),"wb")
                pickle.dump(ce_loss,f)
                f.close()

                if args.optimization == "savlr":
                    f = open(args.save_dir+"/results/condition.pkl", "wb")
                    pickle.dump(condition_d, f)
                    f.close()

    """================"""
    """End ADMM retrain"""
    """================"""

    """================"""
    """ Masked retrain """
    """================"""

    if args.masked_retrain:

        if has_wandb == False:
            retrain_acc = []
            epoch_loss_dict = []

        print("\n!!!!!!!!!!!!!!!!!!! RETRAIN !!!!!!!!!!!!!!!!!")

        # load admm trained model
        print("\n---------------> Loading slr trained file...")

        filename_slr = args.save_dir+"{}/{}_train/cifar10_{}_{}_{}_{}{}.pt".format(args.arch,
                                                                                    args.optimization, 
                                                                                    args.arch,
                                                                                    args.admmtrain_acc, 
                                                                                    args.config_file, 
                                                                                    args.sparsity_type,
                                                                                    args.ext)
        print("!!! Loaded File: ", filename_slr)
        model.load_state_dict(torch.load(filename_slr))
        model.cuda()

        print("\n---------------> Accuracy before hardpruning")
        pred_orig = test(args, model, device, test_loader)
        logging.info('Accuracy before hardpruning: ' + str(float(pred_orig)))
        if has_wandb:
            wandb.log({"retrain_test_acc": pred_orig})
        else:
            retrain_acc.append(pred_orig)

        ADMM = admm.ADMM(args, model, file_name="profile/" + args.config_file + ".yaml", rho=initial_rho)
        admm.hard_prune(args, ADMM, model)
        compression = admm.test_sparsity(args, ADMM, model)
        logging.info('Compression rate: ' + str(compression))

        print("\n---------------> Accuracy after hard-pruning")
        pred_hp = test(args, model, device, test_loader)
        logging.info('Accuracy after hardpruning: ' + str(float(pred_hp)))
        if has_wandb:
            wandb.log({"retrain_test_acc": pred_hp})
        else:
            retrain_acc.append(pred_hp)

        # ------------------
        best_rt = 0
        for epoch in range(1, args.retrain_epoch+1):

            epoch_loss = []
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch * len(train_loader),
                                                             eta_min=4e-08)
            scheduler.step()
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr * (0.5 ** (epoch // lr_decay))

            if args.combine_progressive:
                idx_loss_dict = admm.combined_masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch)
            else:
                idx_loss_dict = admm.masked_retrain(args, ADMM, model, device, train_loader, optimizer, epoch)
            
            prec_rt = test(args, model, device, test_loader)

            if best_rt < prec_rt:
                ## remove old model
                old_file = "cifar10_{}_{}_{}_{}{}.pt".format(args.arch, 
                                                                best_prec1, 
                                                                args.config_file, 
                                                                args.sparsity_type, 
                                                                args.ext)
                if os.path.exists(args.save_dir+args.arch+"/"+args.optimization+"_retrain/"+old_file):
                    os.remove(args.save_dir+args.arch+"/"+args.optimization+"_retrain/"+old_file)
                
                ### save new one
                best_rt = max(prec_rt, best_rt)
                model_best = model
                torch.save(model_best.state_dict(), 
                            args.save_dir+"{}/{}_retrain/cifar10_{}_{}_{}_{}{}.pt".format(args.arch,
                                                                                            args.optimization, 
                                                                                            args.arch,
                                                                                            best_prec1, 
                                                                                            args.config_file, 
                                                                                            args.sparsity_type,
                                                                                            args.ext))
                if has_wandb:
                    wandb.log({"best_retrain_acc": best_rt})
                
            for k, v in idx_loss_dict.items():
                epoch_loss.append(float(v))
            epoch_loss = np.sum(epoch_loss)/len(epoch_loss)

            if has_wandb:
                wandb.log({"losses": epoch_loss[0]})
            else:
                epoch_loss_dict.append(epoch_loss)

        print("---------------> After retraining")
        test(args, model, device, test_loader)
        admm.test_sparsity(args, ADMM, model)

        print("Best Acc: {:.4f}".format(best_prec1))
        logging.info('Best accuracy: ' + str(best_prec1))


    """=============="""
    """masked retrain"""
    """=============="""

            
if __name__ == '__main__':
    main()