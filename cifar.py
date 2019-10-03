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
from kWTA import attack
from kWTA import training
from kWTA import utilities
from kWTA import densenet
from kWTA import resnet
from kWTA import wideresnet


def load_model(model_config):
    modelname = model_config['name']

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

    if 'path' in model_config:
        model.load_state_dict(torch.load(model_config['path']))

    return model

def train(config, cuda_id):

    norm_mean = 0
    norm_var = 1

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
    ])
    cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
    cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(cifar_train, batch_size = config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(cifar_test, batch_size = config['test_batch_size'], shuffle=True)

    eps = config['eps']
    alpha = config['alpha']
    device = torch.device('cuda:{}'.format(cuda_id))

    model = load_model(config['model'])
    
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    # opt = optim.Adam(model.parameters(), lr=config['lr'])

    logfilename = config['logfilename']

    with open(logfilename, 'a+') as logf:
        jstr = json.dumps(config)
        logf.write(jstr+'\n')

    starttime = time.time()

    if 'finetune' in config:
        activation_list = activation.append_activation_list(model, 10000)
        sp = config['finetune']['init_sp']


    for i in range(config['epoch']):
        if 'finetune' in config:
            if i>=config['finetune']['start_epoch']:
                if i%config['finetune']['adjust_epoch'] == 0:
                    sp = sp - config['finetune']['sp_step']
                    sp = round(sp, 5)
                    for l in activation_list:
                        l.sr = sp
        
        if i == config['epoch1']:
            for param_group in opt.param_groups:
                param_group['lr'] = config['epoch1_lr']

        if i == config['epoch2']:
            for param_group in opt.param_groups:
                param_group['lr'] = config['epoch2_lr']

        if 'adv_train' in config:
            if config['adv_train']['attack'] == 'untarg1':
                train_err, train_loss = training.epoch_adversarial(
                    train_loader, model, attack=attack.pgd_linf_untargeted,
                    device=device, opt=opt, num_iter=20, use_tqdm=False, epsilon=eps,
                    randomize=True, alpha=alpha
                )
            elif config['adv_train']['attack'] == 'untarg2':
                train_err, train_loss = training.epoch_adversarial(
                    train_loader, model, attack=attack.pgd_linf_untargeted2,
                    device=device, opt=opt, num_iter=20, use_tqdm=False, epsilon=eps,
                    randomize=True, alpha=alpha
                )
            elif config['adv_train']['attack'] == 'ml':
                train_err, train_loss = training.epoch_adversarial(
                    train_loader, model, attack=attack.pgd_linf_untargeted_mostlikely,
                    device=device, opt=opt, num_iter=20, use_tqdm=False, epsilon=eps,
                    randomize=True, alpha=alpha
                )
            elif config['adv_train']['attack'] == 'trade':
                train_err, train_loss = training.epoch_trade(
                    train_loader, model, opt=opt, device=device, step_size=alpha,
                    epsilon=eps, perturb_steps=10, beta=6
                )
            else:
                raise NotImplementedError
        else:
            train_err, train_loss = training.epoch(train_loader, 
            model, opt, device=device, use_tqdm=False)
        
        test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=False)

        adv_errs = []
        if 'untarg1' in config['test_attack']:
            adv_err, adv_loss = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
            adv_errs.append(adv_err)
        
        if 'untarg2' in config['test_attack']:
            adv_err, adv_loss = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted2, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
            adv_errs.append(adv_err)

        if 'ml' in config['test_attack']:
            adv_err, adv_loss = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted2, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
            adv_errs.append(adv_err)
        

        
        print('epoch: {}'.format(i))
        print('train err: {:.5f}, test err: {:.5f}'.format(train_err, test_err))
        for adv_err in adv_errs:
            print('adv err: {:.5f}'.format(adv_err))
        
        time_e = (time.time()-starttime)/60
        time_r = (config['epoch']-(i+1))*time_e/(i+1)
        print('time elapse: {:.5f} min, time remaining:{:.5f} min'.format(time_e, time_r))
        with open(logfilename, "a+") as logf:
            logf.write('epoch: {}\n'.format(i))
            logf.write('train err: {:.5f}, test err: {:.5f}\n'.format(train_err, test_err))
            for adv_err in adv_errs:
                logf.write('adv err: {:.5f}\n'.format(adv_err))
            logf.write('time elapse: {:.5f} min'.format(time_e))
        torch.save(model.state_dict(), config["savename"])

        if 'finetune' in config:
            print('current sp: {}'.format(sp))
            with open(logfilename, "a+") as logf:
                logf.write('current sp: {}\n'.format(sp))
            torch.save(model.state_dict(), config["finetune"]["savepath"]+"_sp{}.pth".format(sp))
        
        

if __name__ == "__main__":
    # time.sleep(3600*2)
    jsonfile = sys.argv[1]
    cuda_id = int(sys.argv[2])
    print('job:', jsonfile, 'cuda:', cuda_id)
    with open(jsonfile) as jfile:
        config = json.load(jfile)
    for job in config['jobs']:
        print('current job', job)
        train(job, cuda_id)
 