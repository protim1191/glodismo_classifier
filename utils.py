# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:44:26 2022

@author: bhat_po
"""
import os 
from contextlib import redirect_stdout
import pandas as pd
from wavelet import WT
import torch
from torch.utils.data import DataLoader,  random_split
from torchvision import datasets, transforms
import data_new

#%%

class logging:
     def __init__(self, save_path): 
        #training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), training_path))+'\\'
        self.save_path = save_path
        try:
            os.makedirs(save_path)
        except FileExistsError:
            print("")    

     def log(self, logdata):
        with open(self.save_path+r'\alog.txt', 'a') as f:
            with redirect_stdout(f):
                print(logdata)
     def log_metrics(self, results):
        train_logs, val_logs = results
        pd.DataFrame(train_logs).to_csv(os.path.join(self.save_path,'train_metrics.csv'), index=False)
        pd.DataFrame(val_logs).to_csv(os.path.join(self.save_path,'val_metrics.csv'), index=False)

#%% x-lets
def psi_wavelet(x):
  wavelet = WT()
  return wavelet.wt(x.reshape(-1, 1, 28, 28), levels=1).reshape(-1, 784)

def psistar_wavelet(x):
  wavelet = WT()
  return wavelet.iwt(x.reshape(-1, 1, 28, 28), levels=1).reshape(-1, 784)
#%% adversarial generation functions
def generate_fgsm_image(input_image, epsilon, image_grad):
    adversarial_image = input_image + epsilon*image_grad.sign()
    return torch.clamp(adversarial_image, 0, 1) # clipping to [0,1] 
   
#%% load data

def  load_dataset(dataset_preprocess, batch_size, num_workers, dataset = [], data_folder = [], test = False):
    
    if test: # load test data
        if dataset == 'mnist_background_image_tst':
            data = data_new.MNIST_Background_Images_test(data_folder, dataset_preprocess)
        elif dataset == 'mnist_background_random_tst':
            data = data_new.MNIST_Background_Random_test(data_folder, dataset_preprocess)
        elif dataset == 'mnist_rotation_tst':
            data = data_new.MNIST_Rotation_test(data_folder, dataset_preprocess)
        elif dataset == 'mnist_awgn_tst':
            data = data_new.MNIST_AWGN_test(data_folder, dataset_preprocess)
        elif dataset == 'mnist_contrast_awgn_tst':
            data = data_new.MNIST_ContrastAWGN_test(data_folder, dataset_preprocess)
        else:
            if (dataset_preprocess == 'current_norm') or (dataset_preprocess == 'mnist_norm'): 
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
                ])
            else:
                transform=transforms.Compose([transforms.ToTensor()])
            data = datasets.MNIST(r'D:\work_in_progress\PJ_AA_pytorch\data', train=False, transform=transform, target_transform=None, download=True)
            # data = data_new.MNIST_individual_normalized_test()
        
        test_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        return test_loader
    else: #load train data
        if (dataset_preprocess == 'current_norm') or (dataset_preprocess == 'mnist_norm'): 
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform=transforms.Compose([transforms.ToTensor()])
            
        data = datasets.MNIST(r'D:\work_in_progress\PJ_AA_pytorch\data', train=True, download=True,
                            transform=transform)
        
        # data = data_new.MNIST_individual_normalized_train()

        [train, val] = random_split(data, [48000,12000], generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
        return (train_loader, val_loader)

