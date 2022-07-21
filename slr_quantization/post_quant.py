import os
import time
import argparse

import torch
import torch.nn as nn
import numpy as np

from models.lenet_nb_act import Net

import admm as admm
from input_data import MNISTDataLoader as DataLoader


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--load-dir', type=str, default="./checkpoints", metavar='N',
                    help='Directory to save checkpoints')
parser.add_argument('--load-model-name', type=str, default="quantized/quantized_mnist_ternary_acc_99.37.pt", metavar='N',
                    help='For loading the model')
parser.add_argument('--arch', '-a', metavar='ARCH', default='lenet',
                    help='model architecture (default: vgg16_bn)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--num-bits', type=int, default=7, metavar='N',
                    help="If use one side fixed number bits, please set bit length")
args = parser.parse_args()



def save_checkpoint(model, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    state = {}
    state['state_dict'] = model.state_dict()

    torch.save(state, filename)

def quantized_weightbias(x, num_bit=7, use_quant=True):
    if not use_quant:
        return x
    q_num = 2
    level = 2 ** (num_bit - 1 - q_num)
    max_v = 2 ** (q_num - 1)
    min_v = -2 ** (q_num - 1)
    x[x > max_v] = max_v
    x[x < min_v] = min_v

    scale = 1.0 / level
    x = x.div(scale).round().mul(scale)
    return x

def quantized_weightbias2(x, num_bit=7, use_quant=True):
    if not use_quant:
        return x
    hflevel = 2 ** (num_bit-1)
    max_v = x.abs().max()

    scale = (max_v - 0) / hflevel
    x = x.div(scale).round().mul(scale)
    return x



def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        # target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    name_list = []
    if "lenet" in args.arch:
        model = Net()
        print(model)
        print("================================")
        for name, param in model.named_parameters():
            # if param.shape.__len__() is 4 and "shortcut" not in name:
            if "weight" in name:
                name_list.append(name)
        print("Name list:")
        print(name_list)
    use_cuda=True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = DataLoader(args.batch_size, args.batch_size, kwargs)

    test_loader = dataset.test_loader
    criterion = nn.CrossEntropyLoss().cuda()


    model_path = os.path.join(args.load_dir, args.load_model_name)
    print(model_path)
    try:
        model.load_state_dict(torch.load(model_path))
    except:
        try:
            model.load_state_dict(torch.load(model_path)['state_dict'])
        except:
            try:
                model.load_state_dict(torch.load(model_path))
            except:
                raise Exception("Can't load model!")
    model=model.cuda()
    for name, w in model.named_parameters():
        print(name, w.size())
    admm.test_sparsity(model)
    print("Before Quantized:")
    validate(test_loader, model, criterion)

    # for name, w in model.named_parameters():
    #     if name in name_list:
    #         w.data = quantized_weightbias2(w.data, args.num_bits).cuda()
    #
    # print("After Quantized:")
    #
    # acc=validate(test_loader, model, criterion)
    # admm.test_sparsity(model)
    #
    # lay_level_list=[]
    # for name, weight in model.named_parameters():
    #     if name in name_list:
    #         print(name)
    #
    #         unique, counts = np.unique((weight.cpu().detach().numpy()).flatten(), return_counts=True)
    #         un_list = np.asarray((unique, counts)).T
    #         # print("Unique quantized weights counts:\n", un_list)
    #         lay_level_list.append(len(un_list))
    #         print(len(un_list))
    # print(lay_level_list)
    # print(max(lay_level_list))
    # # load_name=args.load_model_name.split("/")[1][:28]
    # load_name=args.load_model_name.split("/")[1][:-3]
    # # save_new_path = args.load_dir +"/quantized/quantized_"+load_name+"{}_crossbar.pt".format(acc)
    # save_new_path = args.load_dir + "/quantized/quantized_" + load_name + "_acc{:.3f}_{}bits_crossbar.pt".format(acc,args.num_bits)
    # save_checkpoint(model, save_new_path)


if __name__ == '__main__':
    main()




