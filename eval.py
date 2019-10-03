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

        model_list.append((model_config['dataset'], modelfile, model))
    return model_list

def load_attacks(attacks_config):
    attack_list = []
    for attack_config in attacks_config['lists']:
        name = attack_config['name']
        attack = {}
        attack['name'] = name
        attack['args'] = {}
        attack['args']['n_test'] = attack_config['n_test']
        attack['args']['epsilon'] = attack_config['epsilon']

        if name == 'FGSM':
            attack['method'] = atk.pgd_linf_untargeted
            attack['foolbox'] = False
            
            attack['args']['alpha'] = attack_config['alpha']
            attack['args']['randomize'] = True
            attack['args']['num_iter'] = 1
        elif name == 'PGD':
            attack['method'] = atk.pgd_linf_untargeted
            attack['foolbox'] = False
            attack['args']['alpha'] = attack_config['alpha']
            attack['args']['num_iter'] = attack_config['num_iter']
            attack['args']['randomize'] = True
        elif name == 'CW':
            attack['method'] = foolbox.attacks.CarliniWagnerL2Attack
            attack['foolbox'] = True
            # attack['args']['d'] = attack_config['d']
            attack['args']['binary_search_steps'] = attack_config['binary_search_steps']
            attack['args']['max_iterations'] = attack_config['max_iterations']
            attack['args']['confidence'] = attack_config['confidence']
            attack['args']['learning_rate'] = attack_config['learning_rate']
            attack['args']['initial_const'] = attack_config['initial_const']
            attack['args']['abort_early'] = True
        elif name == 'deepfool':
            attack['method'] = foolbox.attacks.DeepFoolLinfinityAttack
            attack['foolbox'] = True
            attack['args']['steps'] = attack_config['steps']
            attack['args']['subsample'] = attack_config['subsample']
        elif name == 'boundary':
            attack['method'] = foolbox.attacks.BoundaryAttackPlusPlus
            attack['foolbox'] = True
            # attack['args']['d'] = attack_config['d']
            attack['args']['iterations'] = attack_config['iterations']
            attack['args']['initial_num_evals'] = attack_config['initial_num_evals']
            attack['args']['max_num_evals'] = attack_config['max_num_evals']
        elif name == 'single':
            attack['method'] = foolbox.attacks.SinglePixelAttack
            attack['foolbox'] = True
            attack['args']['max_pixels'] = attack_config['max_pixels']
        elif name == 'local':
            attack['method'] = foolbox.attacks.LocalSearchAttack
            attack['foolbox'] = True
        elif name == 'ALBFGS':
            attack['method'] = foolbox.attacks.ApproximateLBFGSAttack
            attack['foolbox'] = True
            attack['args']['maxiter'] = 20
        elif name == 'pointwise':
            attack['method'] = foolbox.attacks.PointwiseAttack
            attack['foolbox'] = True
        elif name == 'momentum':
            attack['method'] = foolbox.attacks.MomentumIterativeAttack
            attack['foolbox'] = True
            attack['args']['binary_search'] = False
            attack['args']['stepsize'] = 0.007
            attack['args']['iterations'] = 10

        else:
            raise NotImplementedError

        attack_list.append(attack)
    return attack_list

def load_dataset(name):
    if name == 'MNIST':
        dataset = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    elif name == 'CIFAR':
        dataset = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.ToTensor())
    elif name == 'SVHN':
        dataset = datasets.SVHN("./data", split='test', download=True, transform=transforms.ToTensor())
    return dataset

def eval(models_config, attacks_config, log_file, cuda_id):
    model_list = load_model(models_config)
    attack_list = load_attacks(attacks_config)

    device = torch.device("cuda:{}".format(cuda_id))

    for model_ in model_list:
        datasetname, modelfile, model = model_[0], model_[1], model_[2]
        dataset = load_dataset(datasetname)
        with open(log_file, 'a+') as logf:
            logf.write('model: {}\n'.format(modelfile))
        print('model: {}'.format(modelfile))
        model.to(device)
        model.eval()

        loader = DataLoader(dataset, batch_size = 16, shuffle=True)
        err, _ = training.epoch(loader, model, device=device, use_tqdm=True)
        with open(log_file, 'a+') as logf:
            logf.write('standard acc: {}\n'.format(1-err))
        print('standard acc: {}'.format(1-err))
        

        for attack_ in attack_list:
            with open(log_file, 'a+') as logf:
                logf.write(attack_['name']+'\n')
                jstr = json.dumps(attack_['args'])
                logf.write(jstr+'\n')
                print(attack_['name'])
                print(jstr)

            if attack_['foolbox']:
                loader = DataLoader(dataset, batch_size = 1, shuffle=True)
                fmodel = foolbox.models.PyTorchModel(
                    model, bounds=(0, 1), num_classes=10, preprocessing=(0, 1), device=device)
                attack = attack_['method'](fmodel, distance=foolbox.distances.Linfinity)
                err, _ = attack_foolbox.epoch_foolbox(loader, attack, use_tqdm=True, **attack_['args'])
                with open(log_file, 'a+') as logf:
                    logf.write('acc: {}\n'.format(1-err))
                    print('acc: {}'.format(1-err))
            else:
                loader = DataLoader(dataset, batch_size = 16, shuffle=True)
                attack = attack_['method']
                err, _ = training.epoch_adversarial(loader, model, 
                attack, device=device, use_tqdm=True, **attack_['args'])
                with open(log_file, 'a+') as logf:
                    logf.write('acc: {}\n'.format(1-err))
                    print('acc: {}'.format(1-err))
        model.to('cpu')


if __name__ == "__main__":
    json_models = sys.argv[1]
    json_attacks = sys.argv[2]
    cuda_id = int(sys.argv[3])
    log_file = sys.argv[4]


    print('job:', json_models, json_attacks, 'cuda:', cuda_id)
    with open(json_models) as jfile:
        models_config = json.load(jfile)
    with open(json_attacks) as jfile:
        attacks_config = json.load(jfile)

    print(attacks_config, models_config)

    eval(models_config, attacks_config, log_file, cuda_id)

