


import torch
import torch.nn as nn

import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt

from myDatasets import cifar10_dataset

from torchvision.transforms import transforms


def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        
        
def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")


## TO DO ##
def plot_learning_curve(x, y, mode, curve_type, save_path):
    """_summary_
    The function is mainly to show and save the learning curves. 
    input: 
        x: data of x axis 
        y: data of y axis 
    output: None 
    """
    #############
    ### TO DO ### 
    # You can consider the package "matplotlib.pyplot" in this part.

    plt.plot(x, y, label=f'{mode} {curve_type}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{curve_type}')
    plt.title(f'Learning curve of {mode} {curve_type}')
    plt.legend()

    os.makedirs(os.path.join(save_path, f'{mode}'), exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{mode}/{curve_type}.png'))
    plt.close()

def plot_pseudo_labeling(x, y, save_path):
    plt.plot(x, y, label=f'{mode} {curve_type}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{curve_type}')
    plt.title(f'Learning curve of {mode} {curve_type}')
    plt.legend()

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'pseudo_label.png'))
    plt.close()


def train(model, model_name, train_set, unlabeled_set, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer, batch_size):
    start_train = time.time()

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0
    mod = 25
    pseudo_labels = []
    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0

        if i % mod == 0:
            pseudo_set, add_num, flag, pseudo_label = get_pseudo_labels(unlabeled_set, model, batch_size)
            pseudo_labels.append(pseudo_label)
            if flag:
                concat_dataset = torch.utils.data.ConcatDataset([train_set, pseudo_set])
                train_loader_semi = torch.utils.data.DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            else:
                train_loader_semi = train_loader

        # training part
        # start training
        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader_semi)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            loss = criterion(output, label)
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader_semi.dataset)
        train_acc = corr_num / len(train_loader_semi.dataset)
                
        # record the training loss/acc
        overall_loss[i], overall_acc[i] = train_loss, train_acc
        
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0
            corr_num = 0
            val_acc = 0 
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

            valid_loss = []
            valid_acc = []

            for batch in tqdm(val_loader):
                imgs, labels = batch

                logits = model(imgs.to(device))
                loss = criterion(logits, labels.to(device))

                acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

                valid_loss.append(loss.item())
                valid_acc.append(acc)
            
            val_loss = sum(valid_loss) / len(valid_loss)
            val_acc = sum(valid_acc) / len(valid_acc)

            overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc

        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))


    x = range(0,num_epoch)
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()
    # Plot Learning Curve
    ## TO DO ##
    # Consider the function plot_learning_curve(x, y) above
    
    plot_learning_curve(x, overall_acc, 'train', 'acc', f'save_dir/{model_name}')
    plot_learning_curve(x, overall_loss, 'train', 'loss', f'save_dir/{model_name}')
    plot_learning_curve(x, overall_val_acc, 'valid', 'acc', f'save_dir/{model_name}')
    plot_learning_curve(x, overall_val_loss, 'valid', 'loss', f'save_dir/{model_name}')

def get_pseudo_labels(dataset, model, batch_size, threshold=0.99):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    softmax = nn.Softmax(dim=-1)

    pseudo_x = []
    pseudo_y = []
    class_num = 10
    class_count = [0] * class_num
    for i, data in enumerate(data_loader):
        inputs, paths = data

        with torch.no_grad():
            logits = model(inputs.to(device))

        probs = softmax(logits)
        label = torch.where(torch.max(probs, dim=1)[0] > threshold, torch.argmax(probs,dim=1), -1)
        label = label.cpu()
        for j in range(len(label)):
            if label[j] != -1:
                pseudo_x.append(paths[j])
                pseudo_y.append(label[j])
                class_count[label[j]] += 1
    
    print(f"Pseudo Labeling : {len(pseudo_x)}")
    for k in range(class_num):
        print(f'Class Id {k} : {class_count[k]}')

    model.train()
    if len(pseudo_x) > 0:
        flag = 1

        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(30),
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                ])

        pseudo_dataset = cifar10_dataset(pseudo_x, np.array(pseudo_y), train_transform, './p2_data/unlabeled')
        return pseudo_dataset, len(pseudo_x), flag
    else:
        flag = 0
        return None, 0, flag