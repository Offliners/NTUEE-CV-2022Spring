
import torch
import os


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn

from myModels import  myLeNet, myResnet
from myDatasets import  get_cifar10_train_val_set, get_cifar10_unlabeled_set
from tool import train, fixed_seed

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
from cfg import *
import argparse
import sys
from torchvision import models
from torchsummary import summary


def train_interface(args):
    
    """ input argumnet """

    if args.model == 'LeNet':
        model_cfg = LeNet_cfg
    elif args.model == 'myResnet':
        model_cfg = myResnet_cfg
    else:
        print('Unknown Model')
        sys.exit(1)
        

    data_root = model_cfg['data_root']
    model_type = model_cfg['model_type']
    num_out = model_cfg['num_out']
    num_epoch = model_cfg['num_epoch']
    split_ratio = model_cfg['split_ratio']
    seed = model_cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    """ training hyperparameter """
    lr = model_cfg['lr']
    batch_size = model_cfg['batch_size']
    
    ## Modify here if you want to change your model ##

    if model_type == 'LeNet':
        model = myLeNet(num_out=num_out).to(device)
    elif model_type == 'myResnet':
        model = myResnet().to(device)

    # print model's architecture
    print(model)

    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)
    unlabeled_set = get_cifar10_unlabeled_set('./p2_data/unlabeled')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(summary(model.cuda(), (3, 32, 32)))
    
    # define your loss function and optimizer to unpdate the model's parameters.
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.1, epochs=num_epoch, steps_per_epoch=len(train_loader))
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    train(model=model, model_name=model_type, train_set=train_set, unlabeled_set=unlabeled_set, train_loader=train_loader, val_loader=val_loader,
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, batch_size=batch_size)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Model Name', type=str, default='myResnet')
    args = parser.parse_args()
    train_interface(args)