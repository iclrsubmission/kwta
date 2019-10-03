import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import json
import time
import sys
import copy

from kWTA import models
from kWTA import activation
from kWTA import attack as atk
from kWTA import training
from kWTA import utilities
from kWTA import densenet
from kWTA import resnet
from kWTA import wideresnet
from kWTA import attack_foolbox

import foolbox

def load_model(models_config):
    model_list = []
    for model_config in models_config['lists']:
        modelname = model_config['name']
        print(modelname)
        if modelname == 'DenseNet121':
            model = densenet.DenseNet121()
        elif modelname == 'spDenseNet121':
            sp = model_config['sp']
            model = densenet.SparseDenseNet121(sparse_func='vol',
            sparsities=[sp,sp,sp,sp])
        elif modelname == 'ResNet18':
            model = resnet.ResNet18()
        elif modelname == 'ResNet34':
            model = resnet.ResNet34()
        elif modelname == 'ResNet50':
            model = resnet.ResNet50()
        elif modelname == 'ResNet101':
            model = resnet.ResNet101()
        elif modelname == 'ResNet152':
            model = resnet.ResNet152()


        elif modelname == 'spResNet18':
            sp = model_config['sp']
            model = resnet.SparseResNet18(relu=False, sparsities=[sp,sp,sp,sp], sparse_func='vol')
        elif modelname == 'spResNet34':
            sp = model_config['sp']
            model = resnet.SparseResNet34(relu=False, sparsities=[sp,sp,sp,sp], sparse_func='vol')
        elif modelname == 'spResNet50':
            sp = model_config['sp']
            model = resnet.SparseResNet50(relu=False, sparsities=[sp,sp,sp,sp], sparse_func='vol')
        elif modelname == 'spResNet101':
            sp = model_config['sp']
            model = resnet.SparseResNet101(relu=False, sparsities=[sp,sp,sp,sp], sparse_func='vol') 
        elif modelname == 'spResNet152':
            sp = model_config['sp']
            model = resnet.SparseResNet152(relu=False, sparsities=[sp,sp,sp,sp], sparse_func='vol')                
        
        elif modelname == 'spWideResNet':
            sp = model_config['sp']
            model = wideresnet.SparseWideResNet(depth=model_config['depth'], num_classes=10,
            widen_factor=model_config['width'], sp=sp, sp_func='vol', bias=False)
        
        elif modelname == 'WideResNet':
            model = wideresnet.WideResNet(depth=model_config['depth'],
            widen_factor=model_config['width'], num_classes=10)
        else:
            raise NotImplementedError

        model.load_state_dict(torch.load(model_config['path']))
        modelfile = model_config['path'].split('/')[-1][:-4]

        model_list.append((modelfile, model))
    return model_list

def load_dataset(name):
    if name == 'MNIST':
        dataset = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    elif name == 'CIFAR':
        dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    elif name == 'SVHN':
        dataset = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor())
    return dataset

def eval(config, cudaid):
    source_models = load_model(config['source_models'])
    target_models = load_model(config['target_models'])
    dataset = load_dataset(config['dataset'])
    logfilename = config['logfilename']
    device = torch.device("cuda:{}".format(cuda_id))

    if config['dataset'] == 'MNIST':
        eps = 0.3
        alpha = 0.05
    elif config['dataset'] == 'CIFAR':
        eps = 0.031
        alpha = 0.007
    elif config['dataset'] == 'SVHN':
        eps = 0.047
        alpha = 0.01
    else:
        raise NotImplementedError

    test_loader = DataLoader(dataset, batch_size = 25, shuffle=True)

    for (tname ,target) in target_models:
        target.to(device)
        target.eval()
        err, loss = training.epoch(test_loader, target, device=device, use_tqdm=True)
        print('target model: {}'.format(tname))
        print('std acc: {}'.format(1-err))
        with open(logfilename, 'a+') as logf:
            logf.write('target model: {}\n'.format(tname))
            logf.write('std acc: {}\n'.format(1-err))
        for (sname, source) in source_models:
            source.to(device)
            source.eval()
            source_err, err1, err2 = training.epoch_transfer_attack(test_loader,
                                       source, target, 
                                        attack=atk.pgd_linf_untargeted, device=device, n_test=5000, use_tqdm=True,
                                        epsilon=eps,alpha=alpha,num_iter=20,randomize=True)
            print('source model: {}'.format(sname))
            print('adv acc', 1-err1)
            
            with open(logfilename, 'a+') as logf:
                logf.write('source model: {}'.format(sname))
                logf.write('adv acc: {}\n'.format(1-err1))
            source.to('cpu')
        target.to('cpu')
    


if __name__ == "__main__":
    jsonfile = sys.argv[1]
    cuda_id = int(sys.argv[2])
    print('job:', jsonfile, 'cuda:', cuda_id)
    with open(jsonfile) as jfile:
        config = json.load(jfile)
    eval(config, cuda_id)
