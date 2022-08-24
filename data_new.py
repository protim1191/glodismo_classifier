# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:58:51 2022

@author: bhat_po
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms import Compose
#import glob
import numpy as np
#from PIL import Image
#import copy
#import progressbar
import os
from scipy import io as sio

# Tensors are returned, no need to write ToTensor() transform explicitly

class MNIST_original_train_observations(Dataset):
    def __init__(self, num_meas = 78, root_data_folder=r'D:\work_in_progress\PJ_AA_pytorch\data', transforms = None):
        data = datasets.MNIST(root_data_folder, train=True, download=True, transform=None)
        self.labels = data.targets.numpy()
        self.data = data.data.numpy().reshape(60000,784)
        rng = np.random.default_rng(seed = 42)
        A = rng.integers(low=0, high = 2, size=(num_meas,784)).astype('float64')
        for idx in range(A.shape[1]):
            A[:,idx] = A[:,idx]/ np.linalg.norm(A[:,idx])
        
        self.A = A
          
      
    def __len__(self):
         return len(self.labels)
     
    def __getitem__(self, index):
        label = self.data[index,:]
        image = np.dot(self.A,label.T)
        totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
        image = totensor(image)
        return(image,label)
                  
            
            

class MNIST_individual_normalized_train(Dataset):
    def __init__(self, root_data_folder=r'D:\work_in_progress\PJ_AA_pytorch\data', transforms = None):
        data = datasets.MNIST(root_data_folder, train=True, download=True, transform=None)
        self.labels = data.targets.numpy()
        self.data = data.data.numpy().reshape(60000,784)
        data_mean = np.mean(self.data,axis = 1)
        data_std = np.std(self.data,axis = 1)
        self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(60000,28,28)
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        image = self.data[index,:] 
        totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
        image = totensor(image)
        
        return (image,label)
    

class MNIST_individual_normalized_test(Dataset):
    def __init__(self, root_data_folder=r'D:\work_in_progress\PJ_AA_pytorch\data', transforms = None):
        data = datasets.MNIST(root_data_folder, train=False, download=True, transform=None)
        self.labels = data.targets.numpy()
        self.data = data.data.numpy().reshape(10000,784)
        data_mean = np.mean(self.data,axis = 1)
        data_std = np.std(self.data,axis = 1)
        self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(10000,28,28)
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        image = self.data[index,:] 
        totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
        image = totensor(image)
        
        return (image,label)
        

class MNIST_Background_Images_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):    
     
         self.test_data_read_path = os.path.join(root_data_folder, 'mnist_background_images_train.amat')
         self.data = np.loadtxt(self.test_data_read_path)
         self.labels = self.data[:,-1]
         self.transforms = transforms
     
    def __len__(self):
         return len(self.labels)
     
    def __getitem__(self, index):
         label = int(self.labels[index])
         
         image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
         image = self.data[index,:-1].reshape(28,28)
         #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.5455,),(0.2735,)), transforms.ConvertImageDtype(torch.float32)]) 
         #below is the original MNIST normalization
         totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]) 
         image = totensor(image)
         
         if self.transforms:
             image = self.transforms(image)
         
         
        
         return (image,label)
    
class MNIST_Background_Images_test(Dataset):
    def __init__(self, root_data_folder, preprocess, transform = None):    
         self.test_data_read_path = os.path.join(root_data_folder, 'mnist_background_images_test.amat')
         self.data = np.loadtxt(self.test_data_read_path)
         self.labels = self.data[:,-1]
         self.transform = transform
         self.preprocessor = {}
         self.preprocessor = {
             'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.5455,), ((0.2735,))), transforms.ConvertImageDtype(torch.float32)]),
             'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]),          
             'no_norm': Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float32)])
             }
         self.totensor = self.preprocessor[preprocess]
     
    def __len__(self):
         return len(self.labels)
     
    def __getitem__(self, index):
         label = int(self.labels[index])
         
         image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
         image = self.data[index,:-1].reshape(28,28)
         #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.5455,),(0.2735,)), transforms.ConvertImageDtype(torch.float32)]) 
         # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]) 
         image = self.totensor(image) 
        
         if self.transform:
             image = self.transform(image)
         
         
        
         return (image,label) 

class MNIST_Background_Random_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):    
        self.test_data_read_path = os.path.join(root_data_folder, 'mnist_background_random_train.amat')
        self.data = np.loadtxt(self.test_data_read_path)
        self.labels = self.data[:,-1]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        
        image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
        image = self.data[index,:-1].reshape(28,28)
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.5560,),(0.3036,)), transforms.ConvertImageDtype(torch.float32)]) 
        totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]) 
        image = totensor(image)
        
        if self.transforms:
            image = self.transforms(image)
        
        
       
        return (image,label)      
    
class MNIST_Background_Random_test(Dataset):
    def __init__(self, root_data_folder, preprocess, transform = None):    
        self.test_data_read_path = os.path.join(root_data_folder, 'mnist_background_random_test.amat')
        self.data = np.loadtxt(self.test_data_read_path)
        self.labels = self.data[:,-1]
        self.transform = transform
        self.preprocessor = {}
        self.preprocessor = {
            'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.5560,), ((0.3036,))), transforms.ConvertImageDtype(torch.float32)]),
            'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]),          
            'no_norm': Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float32)])
            }
        self.totensor = self.preprocessor[preprocess]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        
        image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
        # totensor = Compose([transforms.ToTensor()]) 
        image = self.data[index,:-1].reshape(28,28)
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.5560,),(0.3036,)), transforms.ConvertImageDtype(torch.float32)])
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)])
        image = self.totensor(image)
        
        if self.transform:
            image = self.transform(image)
        
        return (image,label)      

       
class MNIST_Rotation_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):    
        self.test_data_read_path = os.path.join(root_data_folder, 'mnist_all_rotation_normalized_float_train_valid.amat')
        self.data = np.loadtxt(self.test_data_read_path)
        self.labels = self.data[:,-1]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        
        image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
        # totensor = Compose([transforms.ToTensor()]) 
        image = self.data[index,:-1].reshape(28,28)
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1300,),(0.2970,)),transforms.ConvertImageDtype(torch.float32)])
        totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),transforms.ConvertImageDtype(torch.float32)])
        image = totensor(image)
        
        if self.transforms:
            image = self.transforms(image)
        
        
       
        return (image,label)
    
class MNIST_Rotation_test(Dataset):
    def __init__(self, root_data_folder, preprocess ,transform = None):    
        self.test_data_read_path = os.path.join(root_data_folder, 'mnist_all_rotation_normalized_float_test.amat')
        self.data = np.loadtxt(self.test_data_read_path)
        self.labels = self.data[:,-1]
        self.transform = transform
        self.preprocessor = {}
        self.preprocessor = {
            'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.1300,), ((0.2970,))), transforms.ConvertImageDtype(torch.float32)]),
            'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)]),          
            'no_norm': Compose([transforms.ToTensor(),transforms.ConvertImageDtype(torch.float32)])
            }
        self.totensor = self.preprocessor[preprocess]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(self.labels[index])
        
        image = (255*self.data[index,:-1]).reshape(28,28).astype('uint8')
        # totensor = Compose([transforms.ToTensor()]) 
        image = self.data[index,:-1].reshape(28,28)
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1300,),(0.2970,)), transforms.ConvertImageDtype(torch.float32)])
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)])
        image = self.totensor(image)
        
        if self.transform:
            image = self.transform(image)
        
        
       
        return (image,label)
    

# class MNIST_Rotation_train_normalized_individual(Dataset):
#     def __init__(self, root_data_folder, transforms = None):    
#         self.test_data_read_path = os.path.join(root_data_folder, 'mnist_all_rotation_normalized_float_train_valid.amat')
#         self.data = np.loadtxt(self.test_data_read_path)
#         self.labels = self.data[:,-1]
#         self.data = self.data[:,:-1]
#         data_mean = np.mean(self.data,axis = 1)
#         data_std = np.std(self.data,axis = 1)
#         self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(-1,28,28)
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         label = int(self.labels[index])
#         image = (255*self.data[index,:]).reshape(28,28).astype('uint8')
#         # totensor = Compose([transforms.ToTensor()]) 
#         image = self.data[index,:].reshape(28,28)
#         # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1300,),(0.2970,)),transforms.ConvertImageDtype(torch.float32)])
#         # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),transforms.ConvertImageDtype(torch.float32)]) # MNIST normalization
#         totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
#         image = totensor(image)
        
#         if self.transforms:
#             image = self.transforms(image)
        
        
       
        # return (image,label)


# class MNIST_Rotation_test_normalized_individual(Dataset):
#     def __init__(self, root_data_folder, transforms = None):    
#         self.test_data_read_path = os.path.join(root_data_folder, 'mnist_all_rotation_normalized_float_test.amat')
#         self.data = np.loadtxt(self.test_data_read_path)
#         self.labels = self.data[:,-1]
#         self.data = self.data[:,:-1]
#         data_mean = np.mean(self.data,axis = 1)
#         data_std = np.std(self.data,axis = 1)
#         self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(-1,28,28)
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, index):
#         label = int(self.labels[index])
        
#         image = (255*self.data[index,:]).reshape(28,28).astype('uint8')
#         # totensor = Compose([transforms.ToTensor()]) 
#         image = self.data[index,:].reshape(28,28)
#         # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1300,),(0.2970,)), transforms.ConvertImageDtype(torch.float32)])
#         # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), transforms.ConvertImageDtype(torch.float32)])
#         totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
#         image = totensor(image)
        
#         if self.transforms:
#             image = self.transforms(image)
        
        
       
#         return (image,label)

class MNIST_AWGN_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):
        self.data_read_path = os.path.join(root_data_folder, 'mnist-with-awgn.mat')
        self.data = sio.loadmat(self.data_read_path)
        self.labels = self.data['train_y']
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(np.argmax(self.labels[index]))
        # data = sio.loadmat(self.data_read_path)
        data = self.data['train_x']
        image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.2294,),(0.3054,))]) 
        totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        image = totensor(image)
        
        if self.transforms:
            
            image = self.transforms(image)
       
        
        return (image,label)
    
class MNIST_AWGN_test(Dataset):
   def __init__(self, root_data_folder, preprocess, transform = None):
       self.data_read_path = os.path.join(root_data_folder, 'mnist-with-awgn.mat')
       self.data = sio.loadmat(self.data_read_path)
       self.labels = self.data['test_y']
       self.transform = transform
       self.preprocessor = {}
       self.preprocessor = {
           'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.2294,), (0.3054,))]),
           'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),          
           'no_norm': Compose([transforms.ToTensor()])
           }
       self.totensor = self.preprocessor[preprocess]
   
   def __len__(self):
       return len(self.labels)
   
   def __getitem__(self, index):
       label = int(np.argmax(self.labels[index]))
       # data = sio.loadmat(self.data_read_path)
       data = self.data['test_x']
       image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
       #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.2294,), (0.3054,))]) 
       # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
       image = self.totensor(image)
       
       if self.transform:
           
           image = self.transform(image)
      
       
       return (image,label)
   
# class MNIST_AWGN_test_individual_normalization(Dataset):
#    def __init__(self, root_data_folder, transforms = None):
#        self.data_read_path = os.path.join(root_data_folder, 'mnist-with-awgn.mat')
#        self.data = sio.loadmat(self.data_read_path)
#        self.labels = self.data['test_y']
#        self.data = self.data['test_x']
#        data_mean = np.mean(self.data,axis = 1)
#        data_std = np.std(self.data,axis = 1)
#        self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(-1,28,28)
#        self.transforms = transforms 
   
#    def __len__(self):
#        return len(self.labels)
   
#    def __getitem__(self, index):
#        label = int(np.argmax(self.labels[index]))
#        # data = sio.loadmat(self.data_read_path)
#        #data = self.data['test_x']
#        image = self.data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
#        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.2294,), (0.3054,))]) 
#        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
#        totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
#        image = totensor(image)
       
#        if self.transforms:
           
#            image = self.transforms(image)
      
       
#        return (image,label)
    
   
class MNIST_MotionBlur_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):
        self.data_read_path = os.path.join(root_data_folder, 'mnist-with-motion-blur.mat')
        self.data = sio.loadmat(self.data_read_path)
        self.labels = self.data['train_y']
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(np.argmax(self.labels[index]))
        # data = sio.loadmat(self.data_read_path)
        data = self.data['train_x']
        image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
        
        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1306,), (0.2475,))]) 
        totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        image = totensor(image)
        if self.transforms:
            
            image = self.transforms(image)
       
        
        return (image,label)
    
class MNIST_MotionBlur_test(Dataset):
    def __init__(self, root_data_folder,preprocess, transform = None):
        self.data_read_path = os.path.join(root_data_folder, 'mnist-with-motion-blur.mat')
        self.data = sio.loadmat(self.data_read_path)
        self.labels = self.data['test_y']
        self.transform = transform
        self.preprocessor = {}
        self.preprocessor = {
            'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.1306,), (0.2475,))]),
            'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),          
            'no_norm': Compose([transforms.ToTensor()])
            }
        self.totensor = self.preprocessor[preprocess]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(np.argmax(self.labels[index]))
        # data = sio.loadmat(self.data_read_path)
        data = self.data['test_x']
        image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
        
        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1306,), (0.2475,))]) 
        # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
        image = self.totensor(image)
        
        if self.transform:
            
            image = self.transform(image)
       
        
        return (image,label)
     
    
    


class MNIST_ContrastAWGN_train(Dataset):
    def __init__(self, root_data_folder, transforms = None):
        self.data_read_path = os.path.join(root_data_folder, 'mnist-with-reduced-contrast-and-awgn.mat')
        self.data = sio.loadmat(self.data_read_path)
        self.labels = self.data['train_y']
        self.transforms = transforms 
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = int(np.argmax(self.labels[index]))
        # data = sio.loadmat(self.data_read_path)
        data = self.data['train_x']
        image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
        
        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1505,), (0.2038,))]) # dataset mean and std
        #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # original mnist mean and std
        totensor = Compose([transforms.ToTensor()]) # without normalization, to be used when normalization layer part of network, individual normalization
        image = totensor(image)
        if self.transforms:
            
            image = self.transforms(image)
       
        
        return (image,label)
    
class MNIST_ContrastAWGN_test(Dataset):
  def __init__(self, root_data_folder, preprocess, transform = None):
      self.data_read_path = os.path.join(root_data_folder, 'mnist-with-reduced-contrast-and-awgn.mat')
      self.data = sio.loadmat(self.data_read_path)
      self.labels = self.data['test_y']
      self.transform = transform
      self.preprocessor = {}
      self.preprocessor = {
          'current_norm':  Compose([transforms.ToTensor(), transforms.Normalize((0.1505,), (0.2038,))]),
          'mnist_norm':    Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),          
          'no_norm': Compose([transforms.ToTensor()])
          }
      self.totensor = self.preprocessor[preprocess]
  
  def __len__(self):
      return len(self.labels)
  
  def __getitem__(self, index):
      label = int(np.argmax(self.labels[index]))
      # data = sio.loadmat(self.data_read_path)
      data = self.data['test_x']
      image = data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
      
      #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1505,), (0.2038,))]) 
      # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
      #totensor = Compose([transforms.ToTensor()])
      #image = totensor(image)
      image = self.totensor(image)
      if self.transform:
          
          image = self.transform(image)
     
      return (image,label)
  
# class MNIST_ContrastAWGN_test_individual_normalization(Dataset):
#   def __init__(self, root_data_folder, transforms = None):
#       self.data_read_path = os.path.join(root_data_folder, 'mnist-with-reduced-contrast-and-awgn.mat')
#       self.data = sio.loadmat(self.data_read_path)
#       self.labels = self.data['test_y']
#       self.data = self.data['test_x']
#       data_mean = np.mean(self.data,axis = 1)
#       data_std = np.std(self.data,axis = 1)
#       self.data = ((self.data - np.expand_dims(data_mean,axis = 1))/np.expand_dims(data_std, axis=1)).reshape(-1,28,28)
#       self.transforms = transforms 
  
#   def __len__(self):
#       return len(self.labels)
  
#   def __getitem__(self, index):
#       label = int(np.argmax(self.labels[index]))
#       # data = sio.loadmat(self.data_read_path)
      
#       image = self.data[index,:].reshape(28,28) #resizing to (28,28), train loop written to vectorize
      
#       #totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1505,), (0.2038,))]) 
#       # totensor = Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
#       totensor = Compose([transforms.ToTensor(), transforms.ConvertImageDtype(torch.float32)])
#       image = totensor(image)
#       if self.transforms:
          
#           image = self.transforms(image)
     
#       return (image,label)