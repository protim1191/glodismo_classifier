# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:52:57 2022

@author: bhat_po
"""

from data import MNIST, MNISTWavelet
from recovery import NA_ALISTA, IHT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sensing_matrices import Pixel
from noise import GaussianNoise, StudentNoise, Noiseless
import numpy as np
#from conf import device
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
#%%
def save_log(results, name):
  if len(results) == 2:
    train_logs, test_logs = results
  else:
    test_logs = results
    train_logs = False
  pd.DataFrame(test_logs).to_csv(name + "_test.csv", index=False)
  if train_logs:
    pd.DataFrame(train_logs).to_csv(name + "_train.csv", index=False)
#%% class fro classifier with recovery
class MNIST_CNN_Recovery(nn.Module):
    def __init__(self,batch_size):
        super(MNIST_CNN_Recovery, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.reshape(x,(batch_size,1,28,28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
#%% class for classifier directly after sensor
class MNIST_CNN_WO_Recovery(nn.Module):
    def __init__(self, num_meas, batch_size):
        super(MNIST_CNN_WO_Recovery, self).__init__()
        self.fc0 = nn.Linear(num_meas,784)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc0(x)
        x = torch.reshape(x,(batch_size,1,28,28))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#%%
def train_epoch(sensing_matrix, recovery_model, pred_model , data, noise, use_median, n, positive_threshold, opt, use_mse, train_matrix):
  # train_loss_l2 = 0
  # train_normalizer_l2 = 0
  # train_loss_l1 = 0
  # train_normalizer_l1 = 0
  # false_positives = []
  # false_negatives = []
  train_classifier_loss = 0
  accuracy = 0
  
  for iteration, (X, target) in tqdm(enumerate(iter(data.train_loader))):
  
    X = X.to(device)
    target = target.to(device)
    opt.zero_grad()
    if train_matrix:
      phi = sensing_matrix(X.shape[0])
      forward_op = lambda x: torch.bmm(x.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)
      if not use_median:
        backward_op = lambda x: torch.bmm(x.unsqueeze(1), phi).squeeze(1)
      else:
        backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=False, train_matrix=True)
    else:
      phi = sensing_matrix(1, test=True)
      forward_op = lambda x: torch.matmul(x, phi[0].T)
      if not use_median:    
        backward_op = lambda x: torch.matmul(x, phi[0])
      else:
        backward_op = get_median_backward_op(phi, n, sensing_matrix.d, test=False, train_matrix=False)

      

    y = noise(forward_op(X))
    if recovery_model:
        Xhatt = recovery_model(y, forward_op, backward_op, data.psi, data.psistar)
        Xhat = data.psistar(Xhatt)
        classifier_output = pred_model(Xhat)
    else:
        classifier_output = pred_model(y)

    # true_positives = (torch.abs(X) >= positive_threshold).int()
    # detected_positives = (torch.abs(Xhat) >= positive_threshold).int()

    # if use_mse:
    #   loss = ((Xhat-X)**2).mean()
    # else:
    #   loss = (torch.abs(Xhat-X)).mean()
    
    loss = F.nll_loss(classifier_output, target, reduction='sum')  # sum up batch loss

    loss.backward()
    opt.step()

    # train_normalizer_l2 += (X ** 2).mean().item()
    # train_loss_l2 += ((Xhat-X)**2).mean().item()
    # train_normalizer_l1 += (torch.abs(X)).mean().item()
    # train_loss_l1 += torch.abs(Xhat - X).mean().item()
    train_classifier_loss += loss/data.train_loader.batch_size

    # false_positives.append((detected_positives * (1 - true_positives)).float().mean().item())
    # false_negatives.append((true_positives * (1 - detected_positives)).float().mean().item())
    
    pred = classifier_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    accuracy += pred.eq(target.view_as(pred)).sum().item()
  return {
    # "train_loss_l2": train_loss_l2,
    # "train_loss_l1": train_loss_l1,
    # "train_nmse": 10 * np.log10(train_loss_l2 / train_normalizer_l2),
    # "train_nmae": 10 * np.log10(train_loss_l1 / train_normalizer_l1),
    # "train_false_positives": np.mean(false_positives),
    # "train_false_negatives": np.mean(false_negatives),
    'average_train_accuracy': accuracy/(len(data.train_loader)*data.train_loader.batch_size),
    'average_classifier_loss': train_classifier_loss.item()/len(data.train_loader) 
  }
#%%
def run_experiment(
    n, # length of vectorized image
    sensing_matrix,  
    recovery_model, # recovery algorithm
    pred_model, # classification model
    data,
    use_mse, # sort of redundant as we are using classification losses, e.g. cross entropy ...
    train_matrix,
    use_median,
    noise,
    epochs,
    positive_threshold,
    lr,
    test_model=False,
    ):
         
  # if not test_model: #needs to be figured
  #   test_model = model 
  sensing_matrix = sensing_matrix.to(device)
  if recovery_model:
       recovery_model = recovery_model.to(device)
       
  pred_model = pred_model.to(device)
  
  train_metrics = []
  #test_metrics = [test_epoch(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold)]
  #print("Epoch: 0 Test NMSE:", test_metrics[-1]['test_nmse'],   "Test NMAE:", test_metrics[-1]['test_nmae'])

  "Only train if algorithm or matrix are learnable"
  #if (len(list(model.parameters()))>0 or train_matrix):
    
  learnable_params = list(pred_model.parameters())  
  if recovery_model:  
      learnable_params.extend(list(recovery_model.parameters()))
    
    # if train_matrix:
    #   print('Training Matrix!')
  learnable_params.extend(list(sensing_matrix.parameters()))
  opt = torch.optim.Adam(learnable_params, lr=lr)

  for epoch in range(epochs):
   train_metrics.append(train_epoch(sensing_matrix, recovery_model, pred_model , data, noise, use_median, n, positive_threshold, opt, use_mse, train_matrix))
   data.train_data.reset()
   #test_metrics.append(test_epoch(test_model, sensing_matrix, data, noise, use_median, n, positive_threshold))
   # print("Epoch:", epoch+1, "Test NMSE:", test_metrics[-1]['test_nmse'],   "Test NMAE:", test_metrics[-1]['test_nmae'], "Train NMSE:", train_metrics[-1]['train_nmse'],  "Train NMAE:", train_metrics[-1]['train_nmae'])
   print(f"Train Epoch: {epoch}, Train Accuracy: {train_metrics[-1]['average_train_accuracy']:.3f}, Classifier Loss: {train_metrics[-1]['average_classifier_loss']:.3f}")

  # return train_metrics, test_metrics
  return train_metrics
#%%
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    
    m = 28; n =28;   d =32; random_seed= 50; use_superpixel= 0; initial_scalar = 0.001
    N = m*n # total pixels in image
    meas_per = 0.1 # percentage of measurements in terms of number of pixels,0.1 == 10%
    num_meas = int(meas_per*N)
   # data = MNISTWavelet()
    data = MNIST()
    
    use_recovery = 1
    use_mse = 0
    train_matrix = 1
    use_median = 0
    noise= GaussianNoise(40)
    epochs = 250
    lr = 0.0002
    positive_threshold = 0.01
    s = 50
    sensing_matrix = Pixel(num_meas, N, d, initial_scalar, random_seed, use_superpixel)
    
    batch_size = data.train_loader.batch_size
    
    if use_recovery:
        recovery_model = IHT(15, s)
        pred_model = MNIST_CNN_Recovery(batch_size)
    else: 
        recovery_model = []
        pred_model = MNIST_CNN_WO_Recovery(num_meas,batch_size)
    
    
   
    if recovery_model:
        save_log(run_experiment(
            N, # length of vectorized image
            sensing_matrix,  
            recovery_model, # recovery algorithm
            pred_model, # classification model
            data,
            use_mse, # sort of redundant as we are using classification losses, e.g. cross entropy ...
            train_matrix,
            use_median,
            noise,
            epochs,
            positive_threshold,
            lr,
            test_model=False,
            ),"results/singlepixel_learned_recovery_classifier_"+data.name+"meas_"+str(num_meas))
    else:
        save_log(run_experiment(
            N, # length of vectorized image
            sensing_matrix,  
            recovery_model, # recovery algorithm
            pred_model, # classification model
            data,
            use_mse, # sort of redundant as we are using classification losses, e.g. cross entropy ...
            train_matrix,
            use_median,
            noise,
            epochs,
            positive_threshold,
            lr,
            test_model=False,
            ),"results/singlepixel_learned_no_recovery_classifier_"+data.name+"meas_"+str(num_meas))