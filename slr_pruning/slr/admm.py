from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import operator
import random

import numpy as np
from numpy import linalg as LA
import yaml

try:
    import wandb
    has_wandb = True
except ImportError: 
    has_wandb = False


class ADMM: #ADMM class (but also used for SLR training)
    def __init__(self, args, model, file_name, rho=0.1):
        
        self.ADMM_U = {}
        self.ADMM_Z = {}
        
        self.rho = rho
        self.rhos = {}
        self.conv_wz = {} #convergence behavior ||W-Z||
        self.conv_zz = {} #convergence behavior ||Z(k+1) - Z(k)||
        self.admmloss = {}
        self.ce = 0 #cross entropy loss
        self.ce_prev = 0 #previous cross ent.  loss


        if args.optimization == 'savlr':
            #These parameters are only used in SLR training

            self.W_k = {} #previous W
            self.Z_k = {} #previous Z
            self.W_prev = {} #previous W that satisfied surrogate opt.  condition
            self.Z_prev = {} #previous Z that satisfied surrogate opt.  condition


            self.s = args.initial_s #stepsize
            self.ADMM_Lambda = {} #SLR multiplier
            self.ADMM_Lambda_prev = {} #prev.  slr multiplier
            self.k = 1 #SLR
            self.ce = 0 #cross entropy loss
            self.ce_prev = 0 #previous cross ent.  loss


            """
            This is an array of 1s and 0s. It saves the Surrogate opt. condition after each epoch.

            condition[epoch] = 0/1
            1 means condition was satisfied for that epoch, 0 means condition was not satisfied.
            We will save this array in a pickle file at the end of SLR training to understand the SLR behavior.
            """
            self.condition1 = [] 
            self.condition2 = [] 


        self.init(args, file_name, model)

    def init(self, args, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.safe_load(stream)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                   
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                    if args.optimization == 'savlr':
                        self.W_prev[name] = torch.zeros(W.shape).cuda()
                        self.Z_prev[name] = torch.zeros(W.shape).cuda()
                        self.W_k[name] = W
                        self.Z_k[name] = torch.zeros(W.shape).cuda()
                        self.ADMM_Lambda[name] = torch.zeros(W.shape).cuda()  
                        self.ADMM_Lambda_prev[name] = torch.zeros(W.shape).cuda()  

            except yaml.YAMLError as exc:
                print(exc)

def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")

    
def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(args, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "channel"): 
        shape = weight.shape 
        print("!!! Channel pruning...", weight.shape) 
        weight3d = weight.reshape(shape[0], shape[1], -1) 
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2)) 
        percentile = np.percentile(channel_l2_norm, percent) 
        under_threshold = channel_l2_norm <= percentile 
        above_threshold = channel_l2_norm > percentile 
        weight3d[:,under_threshold,:] = 0 
        above_threshold = above_threshold.astype(np.float32) 
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32) 
        for i in range(weight3d.shape[1]): 
            expand_above_threshold[:, i, :] = above_threshold[i] 
        weight = weight.reshape(shape) 
        expand_above_threshold = expand_above_threshold.reshape(shape) 
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "pattern"):
        # print("pattern pruning...")
        shape = weight.shape

        pattern1 = [[0, 0], [0, 2], [2, 0], [2, 2]]
        pattern2 = [[0, 0], [0, 1], [2, 1], [2, 2]]
        pattern3 = [[0, 0], [0, 1], [2, 0], [2, 1]]
        pattern4 = [[0, 0], [0, 1], [1, 0], [1, 1]]

        pattern5 = [[0, 2], [1, 0], [1, 2], [2, 0]]
        pattern6 = [[0, 0], [1, 0], [1, 2], [2, 2]]
        pattern7 = [[0, 1], [0, 2], [2, 0], [2, 1]]
        pattern8 = [[0, 1], [0, 2], [2, 1], [2, 2]]

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 2]]
        pattern10 = [[0, 0], [0, 2], [1, 0], [1, 2]]
        pattern11 = [[1, 1], [1, 2], [2, 1], [2, 2]]
        pattern12 = [[1, 0], [1, 1], [2, 0], [2, 1]]
        pattern13 = [[0, 1], [0, 2], [1, 1], [1, 2]]

        patterns_dict = {1 : pattern1,
                         2 : pattern2,
                         3 : pattern3,
                         4 : pattern4,
                         5 : pattern5,
                         6 : pattern6,
                         7 : pattern7,
                         8 : pattern8,
                         9 : pattern9,
                         10 : pattern10,
                         11 : pattern11,
                         12 : pattern12,
                         13 : pattern13
                         }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {} # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    
    elif (args.sparsity_type == "random-pattern"):
        print("random_pattern pruning...", weight.shape)
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)

        pattern1 = [0, 2, 6, 8]
        pattern2 = [0, 1, 7, 8]
        pattern3 = [0, 1, 6, 7]
        pattern4 = [0, 1, 3, 4]

        pattern5 = [2, 3, 5, 6]
        pattern6 = [0, 3, 5, 8]
        pattern7 = [1, 2, 6, 7]
        pattern8 = [1, 2, 7, 8]

        pattern9 = [3, 5, 6, 8]
        pattern10 = [0, 2, 3, 5]
        pattern11 = [4, 5, 7, 8]
        pattern12 = [3, 4, 6, 7]
        pattern13 = [1 ,2 ,4, 5]

        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8,
                         9: pattern9,
                         10: pattern10,
                         11: pattern11,
                         12: pattern12,
                         13: pattern13
                         }

        for i in range(shape[0]):
            zero_idx = []
            for j in range(shape[1]):
                pattern_j = np.array(patterns_dict[random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])])
                zero_idx.append(pattern_j + 9 * j)
            zero_idx = np.array(zero_idx)
            zero_idx = zero_idx.reshape(1, -1)
            # print(zero_idx)
            weight2d[i][zero_idx] = 0

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")


def hard_prune(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        # print(name)
        if option == None:
            _, cuda_pruned_weights = weight_pruning(args, W, ADMM.prune_ratios[name])  # get sparse model in cuda

        elif option == "random":
            _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable

def test_sparsity(args, ADMM, model):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    total_zeros = 0
    total_nonzeros = 0
    compression = 0
    if args.sparsity_type == "irregular":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            total_zeros += zeros
            nonzeros = np.sum(W != 0)
            total_nonzeros += nonzeros
            print("Sparsity at layer {} | Weights: {:.0f}, Weights after pruning: {:.0f}, %: {:.3f}, sparsity: {:.3f}".format(name, 
                                                                float(zeros + nonzeros), float(nonzeros), 
                                                                float(nonzeros) / (float(zeros + nonzeros)),
                                                                float(zeros) / (float(zeros + nonzeros))))
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(float(total_weight_number) / float(total_nonzeros)))
        compression = float(total_weight_number) / float(total_nonzeros)
        print("!!!!!!!!!!!! Compression Total| total weights: {:.0f}, total nonzeros: {:.0f}".format(total_weight_number, total_nonzeros))

    elif args.sparsity_type == "column":
        for i, (name, W) in enumerate(model.named_parameters()):

            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            column_l2_norm = LA.norm(W2d, 2, axis=0)
            zero_column = np.sum(column_l2_norm == 0)
            nonzero_column = np.sum(column_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros
    
    elif args.sparsity_type == "channel":

        for i, (name, W) in enumerate(model.named_parameters()):

            if ('bias' in name) or (name not in ADMM.prune_ratios):
                print(name)
                continue
            
            W = W.cpu().detach().numpy()
            shape = W.shape
            W3d = W.reshape(shape[0], shape[1], -1)
            channel_l2_norm = LA.norm(W3d, 2, axis=(0,2))
            zero_row = np.sum(channel_l2_norm == 0)
            nonzero_row = np.sum(channel_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("channel sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros

    elif args.sparsity_type == "filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if 'bias' in name or name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            shape = W.shape
            W2d = W.reshape(shape[0], -1)
            row_l2_norm = LA.norm(W2d, 2, axis=1)
            zero_row = np.sum(row_l2_norm == 0)
            nonzero_row = np.sum(row_l2_norm != 0)
            total_zeros += np.sum(W == 0)
            total_nonzeros += np.sum(W != 0)
            print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
        print('only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
        compression = (total_zeros + total_nonzeros) / total_nonzeros
    
    elif args.sparsity_type == "bn_filter":
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            W = W.cpu().detach().numpy()
            zeros = np.sum(W == 0)
            nonzeros = np.sum(W != 0)
            print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
            compression = zeros / (zeros + nonzeros)
    return compression



def admm_initialization(args, ADMM, model):
    if not args.admm_train:
        return
  
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            # print("!!!!!!!", W.shape)
            _, updated_Z = weight_pruning(args, W, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k) U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z
            ADMM.conv_wz[name] = [] 
            ADMM.conv_zz[name] = []
            ADMM.admmloss[name] = []
         

def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writers):

    # print("!!!!! z_u_update")

    if args.optimization == 'admm':

        if not args.admm_train:
            return

        if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
            for i, (name, W) in enumerate(model.named_parameters()):
                if name not in ADMM.prune_ratios:
                    continue
                Z_prev = None
                Z_prev = ADMM.ADMM_Z[name]
                ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

                _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name],
                                              ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
                ADMM.ADMM_Z[name] = updated_Z
               
                print("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                print("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))

                ADMM.conv_wz[name].extend([float(torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item())]) 
                ADMM.conv_zz[name].extend([float(torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item())])
                

                ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


    elif args.optimization == 'savlr':
        
        # print("!!!!! SLR")

        if not args.admm_train:
            return

        # if batch_idx == 0:
        if (epoch != 1) and (batch_idx == 0):

            print("k = " + str(ADMM.k))

            pow = 1 - (1/(ADMM.k**args.r))
            alpha = 1 - (1/(args.M*(ADMM.k**pow)))
            
            if has_wandb and args.enable_wandb:
                wandb.log({"Hyper/alpha_slr": alpha})

            total_n1 = 0
            total_n2 = 0

            for i, (name, W) in enumerate(model.named_parameters()): 
                #print("at layer : " + name)
                if name not in ADMM.prune_ratios:
                    continue

               
                n1=torch.sqrt(torch.sum((ADMM.W_prev[name] - ADMM.Z_prev[name]) ** 2)).item() #||W(k)-Z(k)||
                n2=torch.sqrt(torch.sum((W - ADMM.Z_prev[name]) ** 2)).item()  #||W(k+1)-Z(k+1)||

                total_n1 += n1
                total_n2 += n2

                ADMM.ADMM_Lambda_prev[name] = ADMM.ADMM_Lambda[name] #save prev.

            satisfied1 = Lagrangian1(ADMM, model) #check if surrogate optimality condition is satisfied.

            if has_wandb and args.enable_wandb:
                wandb.log({"condition/Condition1": satisfied1})
            # else:
            #     condition_d["Condition1"] = condition_d.get("Condition1", [])+satisfied1

            ADMM.condition1.append(satisfied1)

            if satisfied1 == 1 or ADMM.k==1: #if surr. opt. condition is satisfied or k==1
                ADMM.k += 1 #increase k
                if total_n1 != 0 and total_n2 != 0:  #if norms are not 0, update stepsize
                    ADMM.s = alpha * (ADMM.s*total_n1/total_n2) 
                    print("savlr s:")
                    print(ADMM.s)
                    
                    if has_wandb and args.enable_wandb:
                        wandb.log({"Hyper/savlr_s": ADMM.s})

                for i, (name, W) in enumerate(model.named_parameters()):
                    if name not in ADMM.prune_ratios:
                        continue                
                    ADMM.ADMM_Lambda[name] = ADMM.s*(W - ADMM.Z_prev[name]) + ADMM.ADMM_Lambda[name]  #Equation 5 #first update of Lambda

                    ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda[name] #keep the updated lambda

                    ADMM.W_prev[name] = W 
                    ADMM.Z_prev[name] = ADMM.ADMM_Z[name]
                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name] 


            else:
                for i, (name, W) in enumerate(model.named_parameters()):
                    if name not in ADMM.prune_ratios:
                        continue
                    ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda_prev[name] #discard the latest lambda, and save previous lambda
                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name] 


##############################################################################################
            total_n1 = 0
            total_n2 = 0
            for i, (name, W) in enumerate(model.named_parameters()): 
                #print("at layer : " + name)
                if name not in ADMM.prune_ratios:
                    continue

               
                #ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name]

                ADMM.Z_k[name] = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda() #save Z before updated
                ADMM.W_k[name] = W #save W[k] for next epoch
  
                ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name] 
                

                _, updated_Z = weight_pruning(args, ADMM.ADMM_Z[name],
                                              ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
                ADMM.ADMM_Z[name] = updated_Z #Z_(k+1)
                n1=torch.sqrt(torch.sum((ADMM.W_prev[name] - ADMM.Z_prev[name]) ** 2)).item() #||W(k)-Z(k)||
                n2=torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item()  #||W(k+1)-Z(k+1)||

                total_n1 += n1
                total_n2 += n2
              

                #ADMM.conv_wz[name].extend([float(torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item())])  
                #ADMM.conv_zz[name].extend([float(torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-ADMM.Z_k[name])**2)).item())])


                #ADMM.ADMM_Lambda[name] = ADMM.s*(W - ADMM.ADMM_Z[name]) + ADMM.ADMM_Lambda[name]  # Lambda(k+1) = s*(W(k+1) - Z(k+1)) +Lambda(k+0.5) #EQUATION 7
               

            satisfied2 = Lagrangian2(ADMM, model) #check if surrogate optimality condition is satisfied.
            if has_wandb and args.enable_wandb:
                wandb.log({"condition/Condition2": satisfied2})
            # else:
            #     condition_d["Condition2"] = condition_d.get("Condition2", [])+satisfied2

            ADMM.condition2.append(satisfied2)

            if satisfied2 == 1 or ADMM.k==1: #if surr. opt. condition is satisfied or k==1
                print("k = " + str(ADMM.k))

                pow = 1 - (1/(ADMM.k**args.r))
                alpha = 1 - (1/(args.M*(ADMM.k**pow))) 
                
                if has_wandb and args.enable_wandb:
                    wandb.log({"Hyper/alpha_slr": alpha})

                ADMM.k += 1 #increase k

                if total_n1 != 0 and total_n2 != 0:  #if norms are not 0, update stepsize
                    ADMM.s = alpha * (ADMM.s*total_n1/total_n2) 
                    print("savlr s:")
                    print(ADMM.s)
                    
                    if has_wandb and args.enable_wandb:
                        wandb.log({"Hyper/savlr_s": ADMM.s})
                
                for i, (name, W) in enumerate(model.named_parameters()):
                    if name not in ADMM.prune_ratios:
                        continue
                    #ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda[name] #keep the updated lambda
                    ADMM.ADMM_Lambda[name] = ADMM.s*(W - ADMM.ADMM_Z[name]) + ADMM.ADMM_Lambda[name]  # Lambda(k+1) = s*(W(k+1) - Z(k+1)) +Lambda(k+0.5) #EQUATION 7
               
                    ADMM.W_prev[name] = W 
                    ADMM.Z_prev[name] = ADMM.ADMM_Z[name]
                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name] 

            else:
                for i, (name, W) in enumerate(model.named_parameters()):
                    if name not in ADMM.prune_ratios:
                        continue
                    #ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda_prev[name] #discard the latest lambda, and save previous lambda
                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name] 
      


#def z_u_update1(ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writers):
        
#        if not admm_train:
#            return

#        satisfied = 1
#        if epoch != 1 and batch_idx == 0:
            
#            print("k = " + str(ADMM.k))

#            pow = 1 - (1 / (ADMM.k ** r))
#            alpha = 1 - (1 / (M * (ADMM.k ** pow))) 
#            total_n1 = 0
#            total_n2 = 0

#            for i, (name, W) in enumerate(model.named_parameters()): 
#                #print("at layer : " + name)
#                if name not in ADMM.prune_ratios:
#                    continue
    
                
#                n1 = torch.sqrt(torch.sum((ADMM.W_k[name] - ADMM.Z_k[name]) ** 2)).item() #||W(k)-Z(k)||
#                n2 = torch.sqrt(torch.sum((W - ADMM.Z_k[name]) ** 2)).item()  #||W(k+1)-Z(k+1)||

#                total_n1 += n1
#                total_n2 += n2



#                ADMM.ADMM_Lambda_prev[name] = ADMM.ADMM_Lambda[name] #save prev.
#                #ADMM.W_k[name] = W
#                #ADMM.Z_k[name] = ADMM.ADMM_Z[name]
#            satisfied = Lagrangian1(ADMM, model) #check if surrogate optimality condition is satisfied.
#            ADMM.condition1.append(satisfied)

#            if satisfied == 1 or ADMM.k == 1: #if surr.  opt.  condition is satisfied or k==1
#                ADMM.k += 1 #increase k
#                if total_n1 != 0 and total_n2 != 0:  #if norms are not 0, update stepsize
#                    ADMM.s = alpha * (ADMM.s * total_n1 / total_n2) 
#                    print("savlr s:")
#                    print(ADMM.s)

#                for i, (name, W) in enumerate(model.named_parameters()):
#                    if name not in ADMM.prune_ratios:
#                        continue                
#                    ADMM.ADMM_Lambda[name] = ADMM.s * (W - ADMM.Z_prev[name]) + ADMM.ADMM_Lambda[name]  #Equation 5 #first update of Lambda

#                    ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda[name] #keep the updated lambda

#                    ADMM.W_prev[name] = W 
#                    ADMM.Z_prev[name] = ADMM.ADMM_Z[name]
#                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name] / ADMM.rhos[name] 


#            else:
#                for i, (name, W) in enumerate(model.named_parameters()):
#                    if name not in ADMM.prune_ratios:
#                        continue
#                    ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda_prev[name] #discard the latest lambda, and save previous lambda
#                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name] / ADMM.rhos[name] 

#        return satisfied



###############################################################################################
#def z_u_update2(ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writers):
        
#        if not admm_train:
#            return
#        satisfied = 1
#        if epoch != 1 and batch_idx == 0:
#            total_n1 = 0
#            total_n2 = 0
            
#            for i, (name, W) in enumerate(model.named_parameters()): 
#                #print("at layer : " + name)
#                if name not in ADMM.prune_ratios:
#                    continue

               
#                #ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name]/ADMM.rhos[name]

#                ADMM.Z_k[name] = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda() #save Z before updated
#                ADMM.W_k[name] = W #save W[k] for next epoch
  
#                ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name] 
                

#                _, updated_Z = weight_pruning(ADMM.ADMM_Z[name],
#                                              ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
#                ADMM.ADMM_Z[name] = updated_Z #Z_(k+1)
#                n1 = torch.sqrt(torch.sum((ADMM.W_prev[name] - ADMM.Z_prev[name]) ** 2)).item() #||W(k)-Z(k)||
#                n2 = torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item()  #||W(k+1)-Z(k+1)||

#                total_n1 += n1
#                total_n2 += n2
              

#                #ADMM.conv_wz[name].extend([float(torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item())])
#                #ADMM.conv_zz[name].extend([float(torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-ADMM.Z_k[name])**2)).item())])


#                #ADMM.ADMM_Lambda[name] = ADMM.s*(W - ADMM.ADMM_Z[name]) +
#                #ADMM.ADMM_Lambda[name] # Lambda(k+1) = s*(W(k+1) - Z(k+1))
#                #+Lambda(k+0.5) #EQUATION 7
               
#                #ADMM.W_k[name] = W
#                #ADMM.Z_k[name] = ADMM.ADMM_Z[name]
#            satisfied = Lagrangian2(ADMM, model) #check if surrogate optimality condition is satisfied.

#            ADMM.condition2.append(satisfied)

#            if satisfied == 1 or ADMM.k == 1: #if surr.  opt.  condition is satisfied or k==1
#                print("k = " + str(ADMM.k))

#                pow = 1 - (1 / (ADMM.k ** r))
#                alpha = 1 - (1 / (M * (ADMM.k ** pow))) 

#                ADMM.k += 1 #increase k


#                if total_n1 != 0 and total_n2 != 0:  #if norms are not 0, update stepsize
#                    ADMM.s = alpha * (ADMM.s * total_n1 / total_n2) 
#                    print("savlr s:")
#                    print(ADMM.s)

                
#                for i, (name, W) in enumerate(model.named_parameters()):
#                    if name not in ADMM.prune_ratios:
#                        continue
#                    #ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda[name] #keep the
#                    #updated lambda
#                    ADMM.ADMM_Lambda[name] = ADMM.s * (W - ADMM.ADMM_Z[name]) + ADMM.ADMM_Lambda[name]  # Lambda(k+1) = s*(W(k+1) - Z(k+1)) +Lambda(k+0.5) #EQUATION 7
               
#                    ADMM.W_prev[name] = W 
#                    ADMM.Z_prev[name] = ADMM.ADMM_Z[name]
#                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name] / ADMM.rhos[name] 

#            else:
#                for i, (name, W) in enumerate(model.named_parameters()):
#                    if name not in ADMM.prune_ratios:
#                        continue
#                    #ADMM.ADMM_Lambda[name] = ADMM.ADMM_Lambda_prev[name]
#                    ##discard the latest lambda, and save previous lambda
#                    ADMM.ADMM_U[name] = ADMM.ADMM_Lambda[name] / ADMM.rhos[name] 

#        return satisfied
                
def Lagrangian2(ADMM, model):
    '''
    This functions checks Surrogate optimality condition after each epoch. 
    If the condition is satisfied, it returns 1.
    If not, it returns 0.

    '''
    admm_loss = {}
    admm_loss2 = {}
    U_sum = 0 
    U_sum2 = 0
    satisfied = 0 #flag for satisfied condition

    for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue

            U = ADMM.ADMM_Lambda[name] / ADMM.rhos[name]
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + U, p=2) ** 2) #calculate current Lagrangian
            U_sum = U_sum + (0.5 * ADMM.rhos[name] * (torch.norm(U, p=2) ** 2))

            admm_loss2[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.Z_prev[name] + U, p=2) ** 2) #calculate prev Lagrangian
            U_sum2 = U_sum2 + (0.5 * ADMM.rhos[name] * (torch.norm(U, p=2) ** 2))

    Lag = U_sum #current
    Lag2 = U_sum2 #prev

    Lag = ADMM.ce + U_sum #current
    Lag2 = ADMM.ce_prev + U_sum2 #prev

    #print("ce", ADMM.ce)
    #print("ce prev", ADMM.ce_prev)
    for k, v in admm_loss.items():
        Lag += v 
        
    for k, v in admm_loss2.items():
        Lag2 += v 

     
 
    if Lag < Lag2: #if current Lag < previous Lag
        satisfied = 1
        print("condition satisfied")
    else:
        satisfied = 0
        print("condition not satisfied")

    return satisfied

def Lagrangian1(ADMM, model):
    '''
    This functions checks Surrogate optimality condition after each epoch. 
    If the condition is satisfied, it returns 1.
    If not, it returns 0.

    '''
    admm_loss = {}
    admm_loss2 = {}
    U_sum = 0 
    U_sum2 = 0
    satisfied = 0 #flag for satisfied condition

    for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue

            U = ADMM.ADMM_Lambda[name] / ADMM.rhos[name]
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.Z_prev[name] + U, p=2) ** 2) #calculate current Lagrangian
            U_sum = U_sum + (0.5 * ADMM.rhos[name] * (torch.norm(U, p=2) ** 2))

            admm_loss2[name] = 0.5 * ADMM.rhos[name] * (torch.norm(ADMM.W_prev[name] - ADMM.Z_prev[name] + U, p=2) ** 2) #calculate prev Lagrangian
            U_sum2 = U_sum2 + (0.5 * ADMM.rhos[name] * (torch.norm(U, p=2) ** 2))

    Lag = U_sum #current
    Lag2 = U_sum2 #prev

    Lag = ADMM.ce + U_sum #current
    Lag2 = ADMM.ce_prev + U_sum2 #prev

    #print("ce", ADMM.ce)
    #print("ce prev", ADMM.ce_prev)
    for k, v in admm_loss.items():
        Lag += v 
        
    for k, v in admm_loss2.items():
        Lag2 += v 
 
    if Lag < Lag2: #if current Lag < previous Lag
        satisfied = 1
        print("condition satisfied")
    else:
        satisfied = 0
        print("condition not satisfied")

    return satisfied

def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm_train:

        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue
            
            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
            
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_adjust_learning_rate(args, optimizer, epoch):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    
    lr = None
    if epoch % args.admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = epoch % args.admm_epoch

        admm_step = float(args.admm_epoch) / float(3)  # roughly every 1/3 admm_epoch.

        lr = args.lr * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def admm_masked_train(args, ADMM, model, device, train_loader, optimizer, epoch):
    model.train()
    masks = {}
    writer = None

    for i, (name, W) in enumerate(model.named_parameters()):
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    if epoch == 1:
        # inialize Z variable
        # print("Start admm training quantized network, quantization type:
        # {}".format(quant_type))
        admm_initialization(ADMM, model)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        mixed_loss_sum = []
        loss = []
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        ce_loss = F.cross_entropy(output, target)

        z_u_update(ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer)  # update Z and U variables
        ce_loss, admm_loss, mixed_loss = append_admm_loss(ADMM, model, ce_loss)  # append admm losss

        mixed_loss.backward()


        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        mixed_loss_sum.append(float(mixed_loss))
        loss.append(float(ce_loss))

        if batch_idx % args.print_freq == 0:
            print("cross_entropy loss: {}, mixed_loss : {}".format(ce_loss, mixed_loss))
            print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), ce_loss.item()))
            # test_column_sparsity(model)
    lossadmm = []
    for k, v in admm_loss.items():
            print("at layer {}, admm loss is {}".format(k, v))
            lossadmm.append(float(v))

    return lossadmm, mixed_loss_sum, loss

def combined_masked_retrain(args, ADMM, model, device, train_loader, criterion, optimizer, epoch):
    if not masked_retrain:
        return

    idx_loss_dict = {}

    model.train()
    masks = {}

    with open("./profile/" + args.config_file + ".yaml", "r") as stream:
        raw_dict = yaml.safe_load(stream)
        prune_ratios = raw_dict['prune_ratios']
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        _, weight = weight_pruning(args, W, prune_ratios[name])
        W.data = W
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if name in masks:
                W.grad *= masks[name]

        optimizer.step()
        if batch_idx % args.print_freq == 0:
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            print("({}) ({}) cross_entropy loss: {}".format(args.sparsity_type, args.optimizer, loss))
            print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [lr: {}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), current_lr, loss.item()))
        if batch_idx % 10 == 0:
            idx_loss_dict[batch_idx] = loss.item()
    return idx_loss_dict


def masked_retrain(args, ADMM, model, device, train_loader, criterion, optimizer, epoch):
    if not masked_retrain:
        return

    correct = 0
    total = 0
    idx_loss_dict = {}

    model.train()
    masks = {}
    for i, (name, W) in enumerate(model.named_parameters()):
        if name not in ADMM.prune_ratios:
            continue
        above_threshold, W = weight_pruning(args, W, ADMM.prune_ratios[name])
        W.data = W
        masks[name] = above_threshold

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        for i, (name, W) in enumerate(model.named_parameters()):
            if (name in masks) and ("classifier" not in name):
                W.grad *= masks[name]

        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # if batch_idx % args.print_freq == 0:
        #     for param_group in optimizer.param_groups:
        #         current_lr = param_group['lr']
        #     print("({}) cross_entropy loss: {}".format(args.sparsity_type, loss))
        #     print('re-Train Epoch: {} [{}/{} ({:.0f}%)] [{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), current_lr, loss.item()))

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        if batch_idx % args.print_freq == 0:

            print('Retrain Epoch: {}/{} [{}/{} ({:.0f}%)] Loss: {:.6f} | Acc: {:.3f} ({:.0f}/{:.0f})'.format(epoch, args.retrain_epoch,
                                                                                    batch_idx * len(data), len(train_loader.dataset),
                                                                                    100. * batch_idx / len(train_loader), 
                                                                                    loss.item(),
                                                                                    100.*correct/total, correct, total)
                                                                                )
            print("({}) cross_entropy loss: {:.6f}, current lr: {:.6f}".format(args.sparsity_type, loss, current_lr))
        if batch_idx % 1 == 0:
            idx_loss_dict[batch_idx] = loss.item()

    if has_wandb and args.enable_wandb:
        wandb.log({"retrain_train_acc": 100.*correct/total})
    return idx_loss_dict
