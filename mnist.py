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
from kWTA import mnist_model


def train(config, cuda_id):
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size = config['train_batch_size'], shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size = config['test_batch_size'], shuffle=True)

    eps = config['eps']
    alpha = config['alpha']
    device = torch.device('cuda:{}'.format(cuda_id))

    name = config['model']['name']
    if name == 'DNN':
        model = mnist_model.DNN(hidden_size=config['model']['hidden_size'])
    elif name == 'spDNN':
        model = mnist_model.SparseDNN(hidden_size=config['model']['hidden_size'],
        sp=config['model']['sp'], bias=True)
    elif name == 'CNN':
        model = mnist_model.MNIST_CNN(num_channels=config['model']['channels'],
        hidden_size=config['model']['hidden_size'])
    elif name == 'spCNN':
        model = mnist_model.SparseMNIST_CNN(sp1=config['model']['sp1'],
        sp2=config['model']['sp2'], func='vol', num_channels=config['model']['channels'],
        hidden_size=config['model']['hidden_size'])
    else:
        raise ValueError
    
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    logfilename = config['logfilename']

    with open(logfilename, 'a+') as logf:
        jstr = json.dumps(config)
        logf.write(jstr+'\n')

    starttime = time.time()
    for i in range(config['epoch']):
        if config['adv_train']:
            train_err, train_loss = training.epoch_adversarial(
                train_loader, model, attack=attack.pgd_linf_untargeted_mostlikely,
                device=device, opt=opt, num_iter=20, use_tqdm=False, epsilon=eps,
                randomize=True, alpha=alpha
            )
        else:
            train_err, train_loss = training.epoch(train_loader, 
            model, opt, device=device, use_tqdm=False)
        
        test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=False)
        adv_err1, adv_loss1 = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
        use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
        
        adv_err2, adv_loss2 = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted2, device=device, num_iter=20, 
        use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])

        adv_err_ml, adv_loss_ml = training.epoch_adversarial(test_loader,
        model, attack=attack.pgd_linf_untargeted_mostlikely, device=device, num_iter=20, 
        use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])

        
        print('epoch: {}'.format(i))
        print('train err: {}, test err: {}, adv1 err: {}, adv2 err: {}'.format(train_err, test_err, adv_err1, adv_err2))
        print('train err: {}, test err: {}, adv1 err: {}, adv2 err: {}, adv_ml err: {}'.format(train_err, test_err, adv_err1, adv_err2, adv_err_ml))

        time_e = (time.time()-starttime)/60
        time_r = (config['epoch']-(i+1))*time_e/(i+1)
        print('time elapse: {}, time remaining:{}'.format(time_e, time_r))
        with open(logfilename, "a+") as logf:
            logf.write('epoch: {}\n'.format(i))
            logf.write('train err: {}, test err: {}, adv1 err: {}, adv2 err: {}, adv_ml err: {}, time_e:{}min\n'.format(train_err, test_err, adv_err1, adv_err2, adv_err_ml, time_e))
        torch.save(model.state_dict(), config["savename"])
        
    if 'finetune' in config:
        activation_list = activation.append_activation_list(model, 1000)
        opt = optim.SGD(model.parameters(), lr=config['finetune']['lr'], momentum=config['finetune']['momentum'])
        sp = config['model']['sp1']
        for i in range(config['finetune']['epoch']):
            sp = sp - config['finetune']['sp_step']
            for l in activation_list:
                l.sr = sp

            if config['adv_train']:
                train_err, train_loss = training.epoch_adversarial(
                    train_loader, model, attack=attack.pgd_linf_untargeted_mostlikely,
                    device=device, opt=opt, num_iter=20, use_tqdm=False, epsilon=eps,
                    randomize=True, alpha=alpha
                )
            else:
                train_err, train_loss = training.epoch(train_loader, 
                model, opt, device=device, use_tqdm=False)
            
            test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=False)
            adv_err1, adv_loss1 = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
            
            adv_err2, adv_loss2 = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted2, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])

            adv_err_ml, adv_loss_ml = training.epoch_adversarial(test_loader,
            model, attack=attack.pgd_linf_untargeted_mostlikely, device=device, num_iter=20, 
            use_tqdm=False, epsilon=eps, randomize=True, alpha=alpha, n_test=config['n_test_adv'])
            
            print('epoch: {}'.format(i))
            print('current sp: {}'.format(sp))
            print('train err: {}, test err: {}, adv1 err: {}, adv2 err: {}, adv_ml err: {}'.format(train_err, test_err, adv_err1, adv_err2, adv_err_ml))
            
            time_e = (time.time()-starttime)/60
            time_r = (config['finetune']['epoch']-(i+1))*time_e/(i+1)
            print('time elapse: {}, time remaining:{}'.format(time_e, time_r))
            with open(logfilename, "a+") as logf:
                logf.write('epoch: {}\n'.format(i))
                logf.write('current sp: {}'.format(sp))
                logf.write('train err: {}, test err: {}, adv1 err: {}, adv2 err: {}, adv_ml err: {}, time_e:{}min\n'.format(train_err, test_err, adv_err1, adv_err2, adv_err_ml, time_e))
            torch.save(model.state_dict(), config["finetune"]["savepath"]+"_sp{}.pth".format(sp))
        

if __name__ == "__main__":
    # time.sleep(3600*2)
    jsonfile = sys.argv[1]
    cuda_id = int(sys.argv[2])
    print('job:', jsonfile, 'cuda:', cuda_id)
    with open(jsonfile) as jfile:
        config = json.load(jfile)
    # print(config)
    for job in config['jobs']:
        print('current job', job)
        # try:
        train(job, cuda_id)
        # except Exception as e:
        #     print(type(e),e)
        #     print('exception!')