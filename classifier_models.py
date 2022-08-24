# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:59:14 2022

@author: bhat_po
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from recovery import NA_ALISTA, IHT, get_median_backward_op


class mnist_model(nn.Module):
    def __init__(self):
        super(mnist_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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
    

class MNIST_CNN_Recovery(nn.Module):
    def __init__(self,batch_size=16):
        super(MNIST_CNN_Recovery, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.batch_size = batch_size

    def forward(self, x):
        x = torch.reshape(x,(self.batch_size,1,28,28))
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


class MNIST_CNN_WO_Recovery(nn.Module):
    def __init__(self, num_meas, batch_size=16):
        super(MNIST_CNN_WO_Recovery, self).__init__()
        self.fc0 = nn.Linear(num_meas,784)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc0(x)
        x = torch.reshape(x,(self.batch_size,1,28,28))
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
    
class Normalize(nn.Module): # per sample mean-variance normalization.
    def __init__(self):
        super(Normalize, self).__init__()
        
    def forward(self, x): #write in terms of batch inputs
        x_flatten = torch.flatten(x, start_dim=1)
        x_flatten_norm = (x_flatten - x_flatten.mean(axis=1,keepdim=True))/x_flatten.std(axis=1,keepdim=True)
        return torch.reshape(x_flatten_norm,x.shape)

class reshape_layer(nn.Module): # creates a reshape layer
    def __init__(self,batch_size, flatten_shape):
        super(reshape_layer, self).__init__()
        self.resize_shape = [batch_size, flatten_shape]

    def forward(self,x):
        return x.view(self.resize_shape)       
    
class sensor_classifier_model(nn.Module):
    def __init__(self, model_normalize,reshape_layer, sensing_matrix, noise, pred_model):
        super(sensor_classifier_model, self).__init__()
        self.model_normalize = model_normalize #flatten and normalize with sample mean and variance
        self.reshape_layer = reshape_layer # reshape to (batch size, flatten_size) (signal:x)
        self.sensing_matrix = sensing_matrix # apply sensing matrix (y=Ax)
        self.noise = noise  #(y + eta)
        self.pred_model = pred_model # prediction from compressed data.
        
    def forward(self,input_signal,test=False):
        input_signal_normalized = self.model_normalize(input_signal)
        input_signal_normalized_reshaped = self.reshape_layer(input_signal_normalized) # (batch_size, num_elements)
        if not test:
            phi = self.sensing_matrix(input_signal_normalized_reshaped.shape[0])
            forward_op = lambda x: torch.bmm(x.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)
        else:
            phi = self.sensing_matrix(1, test=True)
            forward_op = lambda x: torch.matmul(x, phi[0].T)
        
        meas_wo_noise = forward_op(input_signal_normalized_reshaped)
        y = self.noise(meas_wo_noise)
        classifier_output  = self.pred_model(y)
        return classifier_output   


class sensor_recovery_classifier_model(nn.Module):
    def __init__(self, model_normalize,reshape_layer, sensing_matrix, noise, recovery_model, psi, psistar, pred_model, use_median=0):
        super(sensor_recovery_classifier_model, self).__init__()
        self.model_normalize = model_normalize
        self.reshape_layer = reshape_layer
        self.sensing_matrix = sensing_matrix
        self.noise = noise
        self.recovery_model = recovery_model
        self.psi = psi
        self.psistar = psistar
        self.pred_model = pred_model
        self.use_median = use_median
        
    def forward(self, input_signal,test=False):
        input_signal_normalized = self.model_normalize(input_signal)
        input_signal_normalized_reshaped = self.reshape_layer(input_signal_normalized) # (batch_size, num_elements)
        if not test:
            phi = self.sensing_matrix(input_signal_normalized_reshaped.shape[0])
            forward_op = lambda x: torch.bmm(x.unsqueeze(1), phi.transpose(1, 2)).squeeze(1)
            if not self.use_median:
                backward_op = lambda x: torch.bmm(x.unsqueeze(1), phi).squeeze(1)
            else:
                backward_op = get_median_backward_op(phi, input_signal_normalized_reshaped.shape[1], self.sensing_matrix.d, test=False, train_matrix=True) #input_signal_normalized_reshaped.shape[1]= num elements in image
        else:
            phi = self.sensing_matrix(1, test=True)
            forward_op = lambda x: torch.matmul(x, phi[0].T)
            if not self.use_median:    
                backward_op = lambda x: torch.matmul(x, phi[0])
            else:
                backward_op = get_median_backward_op(phi, input_signal_normalized_reshaped.shape[1], self.sensing_matrix.d, test=False, train_matrix=False)
        
        meas_wo_noise = forward_op(input_signal_normalized_reshaped)
        y = self.noise(meas_wo_noise)
        xr_coeff = self.recovery_model(y, forward_op, backward_op, self.psi, self.psistar)
        xr = self.psistar(xr_coeff)
        classifier_output  = self.pred_model(xr)
        return classifier_output  
        