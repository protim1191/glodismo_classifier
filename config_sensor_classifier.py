# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:19:01 2022

@author: bhat_po
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:30:12 2022

@author: bhat_po
"""


import torch
from datetime import datetime
device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'

data_training = 1

use_adversarial =  0
epsilon = 0.1

dataset_preprocess = 'no_norm' # 'current_norm', 'mnist_norm', 'no_norm'

dataset = [] # [] for original mnist
data_folder = []

# dataset = 'mnist_awgn_tst'
# data_folder =  r'D:\work_in_progress\data\mnist-with-awgn'

# dataset = 'mnist_contrast_awgn_tst'
# data_folder = r'D:\work_in_progress\data\mnist-with-reduced-contrast-and-awgn'

# dataset = 'mnist_rotation_tst'
# data_folder = r'D:\work_in_progress\data\mnist_rotation_new'

# dataset = 'mnist_background_random_tst'
# data_folder = r'D:\work_in_progress\data\mnist_background_random'

# dataset = 'mnist_background_image_tst'
# data_folder = r'D:\work_in_progress\data\mnist_background_images'

load_path =[]
#load_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\train_sensor_classifier_meas_78_d_32_2022-04-09_10-04epoch_200.pt'
# load_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_2022-05-02_22-27\checkpoint_epoch_290.pt'
#load_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_ind_norm_2022-07-22_10-03\checkpoint_epoch_260.pt'
# load_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_ind_norm_2022-08-02_14-07\checkpoint_epoch_10.pt'
#load_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_ind_norm_2022-08-02_14-26\checkpoint_epoch_299.pt'

# if data_training:
#     save_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
# else:
#     save_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_test_'+ str(dataset) + r'_'+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))

if data_training:
    save_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_ind_norm_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
else:
    if use_adversarial:
        save_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_test_ind_norm_adv_'+str(epsilon)+r'_'+ str(dataset) + r'_'+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    else:
         save_path = r'D:\work_in_progress\PJ_AA_pytorch\glodismo_classifier\results\MNIST\only_sensor_classifier_test_ind_norm'+ str(dataset) + r'_'+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))

num_epochs = 300
batch_size = 128
#lr = 0.001 # with recovery loss
lr = 0.0002 #without recovery loss
start_epoch = 0

m = 28  # number of rows in the image
n = 28  # numer of columns in the image
d = 32  # no. of ones per row of the measurement matrix
meas_per = 0.1 # percentage of measurements in terms of number of pixels,0.1 == 10%
use_superpixel= 0
use_median = 0
noise_snr = 40 # in dB
train_val_split = 0.8

random_seed = 50
initial_scalar = 0.001

config_params = {}

config_params = {
    'device': device,
    'dataset': dataset,
    'data_folder': data_folder,
    'data_training' : data_training ,# to train = 1, to test =0
    'batch_size' : batch_size,
    'num_epochs': num_epochs,
    'start_epoch': start_epoch,
    'chkpoint_per_epoch': 10,
    'num_workers': 8,
    'learning_rate': lr,
    'save_path': save_path,
    'load_path': load_path,
    'm':m,
    'n':n,
    'd':d,
    'meas_per':meas_per,
    'use_superpixel':use_superpixel,
    'use_median': use_median,
    'noise_snr': noise_snr,
    'random_seed':random_seed,
    'initial_scalar':initial_scalar,
    'train_val_split ':train_val_split,
    'tr_acc_decrease_epoch': 10,
    'dataset_preprocess': dataset_preprocess,
    'use_adversarial': use_adversarial,
    'epsilon': epsilon
    }

