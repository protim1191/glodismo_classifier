from utils import generate_fgsm_image
import torch

def train_epoch(model, criterion, train_loader, use_median, n, opt, device):
  
  train_classifier_loss = 0.0 # classifier loss
  accuracy = 0 # predicion accuracy
  model.train()
   
  for _, (X, target) in enumerate(train_loader):
         
    X = X.to(device)
    target = target.to(device)
    opt.zero_grad()
    classifier_output = model(X)
    loss_classifier = criterion(classifier_output, target)  # sum up batch loss
    loss_classifier.backward()
    opt.step()

    train_classifier_loss += loss_classifier/train_loader.batch_size
        
    pred = classifier_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    accuracy += pred.eq(target.view_as(pred)).sum()
  
  return {
   
    'average_train_accuracy': accuracy.item()/(len(train_loader)*train_loader.batch_size),
    'average_classifier_loss': train_classifier_loss.item()/len(train_loader) 
   
  } 
#%%
def test_epoch(model, criterion, test_loader,  use_median, n, device, val = 0, use_adversarial = 0 , epsilon = 0):  
  
  adv_correct = 0   # adversarial accuracy
  accuracy = 0   # test prediction accuracy
  
 
  model.eval()
  if use_adversarial:
      for _, (X,target) in enumerate(test_loader):
          X = X.to(device) 
          X.requires_grad = True
          target = target.to(device)
          classifier_output = model(X,test=True)
          pred = classifier_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          accuracy += pred.eq(target.view_as(pred)).sum()
          loss = criterion(classifier_output, target)
          model.zero_grad()
          loss.backward()
          adversarial_input = generate_fgsm_image(X, epsilon, X.grad.data) # generate fgsm adversarial
          adv_classifier_output = model(adversarial_input ,test=True)
          adv_pred = adv_classifier_output.argmax(dim=1,keepdim=True)
          adv_correct += adv_pred.eq(target.view_as(adv_pred)).sum()
          
        
        
        
      return{
        'adv_test_accuracy' : adv_correct.item()/(len(test_loader)*test_loader.batch_size),
        'average_test_accuracy' : accuracy.item()/(len(test_loader)*test_loader.batch_size)
        }
              
              
              
            
          
  else:
    
      with torch.no_grad():
        if val:
           val_total_loss = 0.0
            
        for _, (X, target) in enumerate(test_loader):
          X = X.to(device) 
          target = target.to(device)
          classifier_output = model(X,test=True)
          pred = classifier_output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          accuracy += pred.eq(target.view_as(pred)).sum()
          if val:
              loss_nll = criterion(classifier_output, target) 
              val_total_loss += loss_nll/test_loader.batch_size
        if val:
            return {
                'average_val_accuracy': accuracy.item()/(len(test_loader)*test_loader.batch_size),
                'average_val_loss': val_total_loss.item()/len(test_loader)
                }
        return {
          'average_test_accuracy': accuracy.item()/(len(test_loader)*test_loader.batch_size),
          }

