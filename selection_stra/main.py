import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from data import get_mnist, get_cifar10, get_cifar100, CLDataLoader
from torch.utils.data import DataLoader
from pydoc import locate
from data_select import data_selection
from architect import Architect
from model import MLP, Resnet18
import utils
from torch.nn import functional as F
import logging
import sys
import os
import time
import glob

def random_selection_data(dataset, select_num):
    select_id = torch.randperm(len(dataset))[:select_num]
    return dataset[select_id]

def Train(train_loader, val_data, model, args, architect, weight_arch, network_optimizer):
    model.train()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for step, (train_input, train_target, train_id) in enumerate(train_loader):
        val_input, val_target, _ = random_selection_data(val_data, select_num = args.select_num)
        if torch.cuda.is_available():
            train_input, train_target = train_input.cuda(), train_target.cuda()
            val_input, val_target = val_input.cuda(), val_target.cuda()

        architect.step(
            train_input, train_target, val_input, val_target, train_id, args.learning_rate, network_optimizer, args.unrolled)

        network_optimizer.zero_grad()
        logits = model(train_input)
        loss = F.cross_entropy(logits, train_target, reduction = 'none')
        weight = weight_arch.get_weight(train_input, train_id)
        loss = (weight * loss).mean()
        loss.backward()
        network_optimizer.step()


        n = train_input.size(0)
        prec1, prec5 = utils.accuracy(logits, train_target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def infer(valid_queue, model):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target, _) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = F.cross_entropy(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

parser = argparse.ArgumentParser()

#----------------for experiment------------------
parser.add_argument('--n_epochs', type = int, default = 10)
parser.add_argument('--select_num', type = int, default = 64)
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--batch_size', type = int, default = 64)

#----------------for data-------------------
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mnist', choices=[
        'mnist', 'cifar10', 'cifar100'
])
parser.add_argument('--data_path', type = str, default='none')
parser.add_argument('--n_tasks', type = int, default=1)
parser.add_argument('--n_classes', type = int, default=10)
parser.add_argument('--input_size', type=list, default=[1, 28, 28])

# ---------------for parameter----------------
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

# ---------------for arch parameter----------------
parser.add_argument('--unrolled', action='store_false', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

args = parser.parse_args()



args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    data = locate('get_{}'.format(args.dataset))(args)
    train_data, val_data, test_data = data

    if args.dataset in 'mnist':
        model = MLP(args)
    elif args.dataset in 'cifar10' or args.dataset in 'cifar100':
        model = Resnet18(args)
    else:
        raise Exception('error')
    weight_arch = data_selection(data[0])
    architect = Architect(model, weight_arch, args)

    train_loader  = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(val_data, batch_size = 64, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_data, batch_size = 64, shuffle = True, drop_last = False)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    print(optimizer.state)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.n_epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.n_epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj = Train(train_loader, val_data, model, args, architect, weight_arch, optimizer)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(val_loader, model)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


if __name__ == '__main__':
    main()
