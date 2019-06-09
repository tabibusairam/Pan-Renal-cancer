from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F     
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

use_gpu = torch.cuda.is_available()

# from sampler import ImbalancedDatasetSampler

##################################################################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_conv = torchvision.models.resnet18(pretrained=True)
#print(model_conv)
#num_ftrs = model_conv.fc.in_features

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)


print(model_conv)
#print('-'*50)
#params = list(model_conv.parameters())
#print(params)
model_conv = model_conv.to(device)
#model_conv.load_state_dict(torch.load('/shared/supernova/home/sairam.tabibu/results/kidney/subtype/resnet-18-subtype_classification'))

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.00005, weight_decay=0.05)
exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_conv, 'min', patience=2, verbose=True, factor = 0.2)
#soft = nn.Softmax()


#################################################################################################################

data_dir = '/tmp/kidney'
data_dir1 = '/tmp/kidney'
resize = [224,224]
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
	        #transforms.RandomRotation(45),
            #transforms.RandomRotation(30),
            #transforms.RandomRotation(15),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.583], [0.2066, 0.240, 0.186]),    
        #transforms.TenCrop(max(resize)),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.620, 0.446, 0.594], [0.218, 0.248, 0.193])(crop) for crop in crops])),
        ]),
    }



####################################################################################################################

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}


image_datasets1 = {x: datasets.ImageFolder(os.path.join(data_dir1, x),
                                          data_transforms[x])
                  for x in ['valid']}




# If you want to use weighted sampling 

# class_counts = [213302,74626,202912]
# weights = 1./torch.tensor(class_counts, dtype=torch.float)

# target = torch.cat((torch.zeros(class_counts[0], dtype=torch.long),torch.ones(class_counts[1], dtype=torch.long),torch.ones(class_counts[2], dtype=torch.long) * 2))

# print('target train 0/1/2: {}/{}/{}'.format(
#     (target == 0).sum(), (target == 1).sum(), (target == 2).sum()))

# class_sample_count = torch.tensor(
#     [(target == t).sum() for t in torch.unique(target, sorted=True)])
# weight = 1. / class_sample_count.float()
# samples_weight = torch.tensor([weight[t] for t in target])

# # Create sampler, dataset, loader
# sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128)
              for x in ['train']}


dataloaders1 = {x: torch.utils.data.DataLoader(image_datasets1[x], batch_size=125, shuffle = False)
              for x in ['valid']}       


dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}

dataset_sizes1 = {x: len(image_datasets1[x]) for x in ['valid']}

class_names = image_datasets['train'].classes





#############################################################################################################


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_error=2.0
    best_model_wts1 = model.state_dict()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and vali dation phase
        for phase in ['train','valid']:
            
            k=0;
            running_loss = 0.0
            running_corrects = 0



#################################################################################################################
            if(phase=='train'):
                model.train()
                print(phase)
                print(time.time()-since)
                for inputs, labels in dataloaders[phase]:
                    k +=1
                    if(k%100 ==0):
                    	print(k/100)
                  #  		f=open("/shared/supernova/home/sairam.tabibu/training/status_18.txt" , "a+")
                		# f.write(phase+"-"+str(k/100))
                		# f.write("\n")
                		# f.close()
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    #print(time.time() - since)
                    
                    
                    # labels1 = labels.unsqueeze(1)
                    # labels_onehot = torch.FloatTensor(labels.size()[0], 2)
                    # labels_onehot.zero_()
                    # labels_onehot.scatter_(1, labels1, 1)
                    

                    # if use_gpu:
                    #     inputs = Variable(inputs.cuda())
                    #     labels = Variable(labels.cuda())
                    #     # labels_onehot = Variable(labels_onehot.cuda())

                    # else:
                    #     inputs = Variable(inputs)
                    #     labels = Variable(labels)
                        # labels_onehot = Variable(labels_onehot)

                    optimizer.zero_grad()
                    

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                    #outputs = soft(outputs)
                    # outputs = F.tanh(outputs)
                    # outputs = (outputs+1)/2
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)


                    #computing loss

                    loss.backward()
                    optimizer.step()
                    #print(time.time() - since)

                    f=open("/shared/supernova/home/sairam.tabibu/training/res-tr.txt" , "a+")
                    f.write(str(loss.item()))
                    f.write("\n")
                    f.close()

                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = (running_loss*1.0) / dataset_sizes[phase]
                epoch_acc = (running_corrects.double()) / dataset_sizes[phase]
                    
                f=open("/shared/supernova/home/sairam.tabibu/training/resnet-tr-loss.txt" , "a+")
                f.write(str(epoch_loss))
                f.write("\n")
                f.close()
                f=open("/shared/supernova/home/sairam.tabibu/training/resnet-tr-acc.txt" , "a+")
                f.write(str(epoch_acc))
                f.write("\n")
                f.close()

##############################################################################################################  

            if(phase == 'valid'):
                
                model.eval()
                print(phase)
                cn = 0
                run_cor = 0    

                for inputs, labels in dataloaders1[phase]:

                    k +=1
                    if(k%100 ==0):
                    	print(k/100)
                    f=open("/shared/supernova/home/sairam.tabibu/training/status_18.txt" , "a+")
                    f.write(phase+"-"+str(k/100))
                    f.write("\n")
                    f.close()
                # get the inputs
                    inputs = inputs.to(device)
                    labels = labels.to(device)


                    # labels1 = labels.unsqueeze(1)
                    # labels_onehot = torch.FloatTensor(labels.size()[0], 2)
                    # labels_onehot.zero_()
                    # labels_onehot.scatter_(1, labels1, 1)
                # wrap them in Variable
                    # if use_gpu:
                    #     inputs = Variable(inputs.cuda(), volatile=True)
                    #     labels = Variable(labels.cuda(), volatile=True)
                    #     #labels_onehot = Variable(labels_onehot.cuda(), volatile=True)
                    # else:
                    #     inputs = Variable(inputs)
                    #     labels = Variable(labels)
                        #labels_onehot = Variable(labels_onehot)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):

                    #bs, ncrops, c, h, w = inputs.size()
                    #inp_chg = inputs.view(-1, c, h, w)
                        outputs = model(inputs) 
                    #outputs = F.tanh(outputs)
                    #outputs = (outputs+1)/2
                    
                    #outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)

                    f=open("/shared/supernova/home/sairam.tabibu/training/res-vl.txt" , "a+")
                    f.write(str(loss.item()))
                    f.write("\n")
                    f.close()

                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if(cn==16):
                        f=open("/shared/supernova/home/sairam.tabibu/training/validation-class" , "a+")
                        f.write(str(run_cor))
                        f.write("\n")
                        f.close()
                        run_cor = 0
                        cn=0

                    if(cn<16):
                        run_cor += torch.sum(preds == labels.data)
                        cn = cn + 1


                
                epoch_loss = (running_loss*1.0) / dataset_sizes1[phase]
                epoch_acc = (running_corrects.double()) / dataset_sizes1[phase]

                scheduler.step(epoch_loss)
                f=open("/shared/supernova/home/sairam.tabibu/training/resnet-vl-loss.txt" , "a+")
                f.write(str(epoch_loss))
                f.write("\n")
                f.close()
                f=open("/shared/supernova/home/sairam.tabibu/training/resnet-vl-acc.txt" , "a+")
                f.write(str(epoch_acc))
                f.write("\n")
                f.close()    

##############################################################################################################

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase =='valid' and epoch_loss < best_error:
                best_error = epoch_loss
                best_model_wts1 = copy.deepcopy(model.state_dict())

            
            print('Best val Acc: {:4f}'.format(best_acc))
            print('Least val Err: {:4f}'.format(best_error))
            f=open("/shared/supernova/home/sairam.tabibu/training/status_18.txt" , "a+")
            f.write("best acc - " + str(best_acc))
            f.write("\n")
            f.write("least acc - " + str(best_error))
            f.write("\n")
            f.close()    
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts,'/shared/supernova/home/sairam.tabibu/training/resnet_wei.pth')
    torch.save(best_model_wts1,'/shared/supernova/home/sairam.tabibu/training/resnet_wei1.pth')
    return model


#################################################################################################################


    


##################################################################################################################

model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
                       num_epochs=10)


