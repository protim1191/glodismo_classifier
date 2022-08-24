# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 19:08:16 2022

This script learns both the sensor and the classifier. Change parameters in the config_sensor_classifier.py

@author: protim bhattacharjee 
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 11:53:06 2022

@author: bhat_po
"""


import torch
import torch.nn as nn
import torch.optim as optim
from sensing_matrices import Pixel
from noise import GaussianNoise#, StudentNoise, Noiseless
import time
import config_sensor_classifier as config
#from recovery import get_median_backward_op
from classifier_models import  MNIST_CNN_WO_Recovery, Normalize, reshape_layer, sensor_classifier_model
from utils import logging, load_dataset
from train_test_routines import train_epoch, test_epoch
#from torchsummary import summary
#import os
#from contextlib import redirect_stdout

  
#%%
if __name__=='__main__':
    
    save_path = config.config_params['save_path']
    logger = logging(save_path)
    logger.log(config.config_params)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = config.config_params['device']
    logger.log(device)
    print(device)
    
    
    
    batch_size = config.config_params['batch_size']
    num_epochs = config.config_params['num_epochs']
    start_epoch = config.config_params['start_epoch']
    chkpoint_per_epoch = config.config_params['chkpoint_per_epoch']
    num_workers = config.config_params['num_workers']
    lr = config.config_params['learning_rate']
    dataset = config.config_params['dataset']
    data_training = config.config_params['data_training']
    data_folder = config.config_params['data_folder']
    load_path = config.config_params['load_path']
    use_adversarial = config.config_params['use_adversarial']
    epsilon = config.config_params['epsilon']
    m = config.config_params['m']   
    n = config.config_params['n']
    d = config.config_params['d']
    meas_per = config.config_params['meas_per']
    use_superpixel=config.config_params['use_superpixel']
    use_median = config.config_params['use_median']
    noise_snr = config.config_params['noise_snr']
    random_seed= config.config_params['random_seed']
    initial_scalar = config.config_params['initial_scalar']
    tr_acc_decrease_flag = 1 # for detecting decrease in trainin accuracy, callback not implemented.
    tr_acc_decrease_epoch = config.config_params['tr_acc_decrease_epoch']
    dataset_preprocess = config.config_params['dataset_preprocess']
   
    
#%% load data    
    if data_training:
        train_loader, val_loader = load_dataset(dataset_preprocess, batch_size=batch_size, num_workers = num_workers, dataset = [], data_folder = data_folder, test = False)
    else:
        test_loader = load_dataset(dataset_preprocess, batch_size=batch_size, num_workers = num_workers, dataset = [], data_folder = data_folder, test = True)
        
#%% model initilaization    
    
    N = m*n # total pixels in image
    num_meas = int(meas_per*N)
    noise= GaussianNoise(noise_snr)
    
    model_normalize = Normalize()
    reshp_layer = reshape_layer(batch_size, N)
    sensing_matrix = Pixel(num_meas, N, d, initial_scalar, random_seed, use_superpixel)
    pred_model = MNIST_CNN_WO_Recovery(num_meas, batch_size)
    
    sensor_classifier = sensor_classifier_model(model_normalize, reshp_layer, sensing_matrix, noise, pred_model)
    
     
    logger.log(model_normalize)
    logger.log(sensing_matrix)
    logger.log(pred_model)
        
    sensor_classifier.to(device)
    criterion = nn.CrossEntropyLoss()
    
    learnable_params = list(sensor_classifier.pred_model.parameters())  
    learnable_params.extend(list(sensor_classifier.sensing_matrix.parameters()))
    opt =  optim.Adam(learnable_params, lr=lr)
    
    if load_path:
        checkpoint = torch.load(load_path)
        sensor_classifier.sensing_matrix.load_state_dict(checkpoint['sensor_state_dict'])
        sensor_classifier.pred_model.load_state_dict(checkpoint['pred_model'])
        opt.load_state_dict(checkpoint['opt_state'])
#%% training and testing loop    
    if data_training:
        #training loop
        total_train_time = 0
        untrained_val_metric = test_epoch(sensor_classifier, criterion, val_loader,  use_median, n, device,val = 1) # Try untrained model
        if start_epoch > 0:
            logger.log(f'restarting training at epoch:{start_epoch}' )
            print(f'restarting training at epoch:{start_epoch}')
        logger.log(f'untrained validation loss: {untrained_val_metric["average_val_loss"]:.5f}, untrained validation accuracy: {untrained_val_metric["average_val_accuracy"]:.5f}')
        print(f'untrained validation loss: {untrained_val_metric["average_val_loss"]:.5f}, untrained validation accuracy: {untrained_val_metric["average_val_accuracy"]:.5f}')
        
        train_metrics = []
        val_metrics = []
        best_train_acc = 0.0
        best_train_epoch = 0
        best_val_acc = 0.0
        best_val_epoch = 0
        
        # start training
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            train_metrics.append(train_epoch(sensor_classifier, criterion, train_loader,  use_median, n, opt, device))
            val_metrics.append(test_epoch(sensor_classifier, criterion, val_loader,  use_median, n, device,val = 1))
            epoch_time = time.time()-start_time
            total_train_time += epoch_time
            print(f"Train Epoch: {epoch}, Train Accuracy: {train_metrics[-1]['average_train_accuracy']:.5f}, Classifier Loss: {train_metrics[-1]['average_classifier_loss']:.5f}, Validation accuracy: {val_metrics[-1]['average_val_accuracy']:.5f}, Validation loss: {val_metrics[-1]['average_val_loss']:.5f}, Epoch Time:{epoch_time:.3f}s")
            logger.log(f"Train Epoch: {epoch}, Train Accuracy: {train_metrics[-1]['average_train_accuracy']:.5f}, Classifier Loss: {train_metrics[-1]['average_classifier_loss']:.5f}, Validation accuracy: {val_metrics[-1]['average_val_accuracy']:.5f}, Validation loss: {val_metrics[-1]['average_val_loss']:.5f}, Epoch Time:{epoch_time:.3f}s")
            if train_metrics[-1]['average_train_accuracy'] > best_train_acc:
                best_train_acc = train_metrics[-1]['average_train_accuracy']
                best_train_epoch = epoch
            if val_metrics[-1]['average_val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics[-1]['average_val_accuracy']
                best_val_epoch = epoch
            if epoch%chkpoint_per_epoch == 0 or epoch == num_epochs - 1:
                 torch.save({ 'sensor_state_dict': sensor_classifier.sensing_matrix.state_dict(),
                             'pred_model': sensor_classifier.pred_model.state_dict(),
                             'opt_state': opt.state_dict()
                     }, save_path + r'\checkpoint_epoch_'+str(epoch)+'.pt')
            
            # if tr_acc_decrease_flag and epoch > 1:
            #  if train_metrics[-1]['average_train_accuracy'] < train_metrics[-2]['average_train_accuracy'] :
            #        logger.log('training accuracy decreased!')
            #        print('training accuracy decreased!')
            #        tr_acc_decrease_flag+= 1
            #        if tr_acc_decrease_flag >= tr_acc_decrease_epoch:
            #            print('Consecutive decrease in training accuracy for {tr_acc_decrease_epoch} iterations, training aborted.')
            #            torch.save({ 'sensor_state_dict': sensing_matrix.state_dict(),
            #                        'pred_model': pred_model.state_dict(),
            #                        'recovery_model': recovery_model.state_dict(),
            #                        'opt_state': opt.state_dict()
            #                }, save_path + r'\checkpoint_epoch_'+str(epoch)+'.pt')
            #            break
            #  else:
            #     tr_acc_decrease_flag = 0
            
        results = [train_metrics, val_metrics]
        logger.log_metrics(results)
        logger.log(f'Total training time: {total_train_time} s')
        logger.log(f'best train accuracy:{best_train_acc:.5f} at epoch: {best_train_epoch }' )
        logger.log(f'best validation accuracy:{best_val_acc:.5f} at epoch: {best_val_epoch }' )
        print(f'Total training time: {total_train_time} s')
        print(f'best train accuracy:{best_train_acc:.5f} at epoch: {best_train_epoch }')
        print(f'best validation accuracy:{best_val_acc:.5f} at epoch: {best_val_epoch}')
        
    else:
        logger.log(f'Testing on {dataset}')
        print(f'Testing on {dataset}')
        test_metric = test_epoch(sensor_classifier, criterion, test_loader, use_median, n, device,val = 0, use_adversarial=use_adversarial, epsilon = epsilon)    
        logger.log(f'Test Accuracy: {test_metric["average_test_accuracy"]:.5f}')
        print(f'Test Accuracy: {test_metric["average_test_accuracy"]:.5f}')
        if use_adversarial:
            logger.log(f'Adversarial Test Accuracy with epsilon {epsilon}:{test_metric["adv_test_accuracy"]:.5f}')
            print(f'Adversarial Test Accuracy with epsilon {epsilon}: {test_metric["adv_test_accuracy"]:.5f}')
        
  
