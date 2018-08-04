from __future__ import print_function, division
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


use_gpu = torch.cuda.is_available()



#################################################


model_conv = torchvision.models.inception_v3(pretrained=True)
#print(model_conv)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

print(model_conv)
#print('-'*50)
#params = list(model_conv.parameters())
#print(params)
if use_gpu:
	model_conv = nn.DataParallel(model_conv).cuda()


criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#################################################

data_dir = '/Neutron3/Datasets/med_img'
#data_dir = '/Users/user/Desktop/medical_images'
resize = [299,299]
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(max(resize)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            #Higher scale-up for inception
            transforms.Resize(int(max(resize))),
            #transforms.CenterCrop(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
											 shuffle=True)
			  for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes




###################################################t


def train_model(model, criterion, optimizer, scheduler, num_epochs):
	since = time.time()

	best_model_wts = model.state_dict()
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		f = open("losses.txt","a+")
		f.write("epoch number  " + str(epoch) + "\n" )
		f.close()	
		
		f=open("losses1.txt","a+")
		f.write("epoch number  " + str(epoch) + "\n" )
		f.close()		
		# Each epoch has a training and vali dation phase 'train',
		for phase in ['train', 'test']:
			print(phase)
			if phase == 'train':
				scheduler.step()
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
			k=0;
			# Iterate over data.
			for data in dataloaders[phase]:
				k=k+1;
				if(k%100 == 0):
					print(k/100)
					
				# get the inputs
				inputs, labels = data

				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				#print(inputs.size())
				#res = model(inputs)
				#print(res.size())
				if phase == 'train':
					outputs,aux_outputs = model(inputs)	
					#print(outputs.size())
				else:
					outputs = model(inputs)	
					#print(outputs.size())
				#print(outputs1)
				#outputs = torch.FloatTensor(outputs1)
				#print("-----------------")
				#print(outputs)
				_, preds = torch.max(outputs.data, 1)
				#print("-----------------")
				#print(preds)
				if phase =='train':
					loss1 = criterion(outputs, labels)
					loss2 = criterion(aux_outputs, labels)
					loss = loss1 + 0.4*loss2
				else:
					loss = criterion(outputs, labels)
				#print(loss)	
				if phase == 'train':
					f=open("losses.txt" , "a+")
					f.write(str(loss))
					f.write("\n")
					f.close() 
				
				if phase == 'test':
					f=open("losses1.txt","a+") 
					f.write(str(loss))
					f.write("\n")
					f.close()
					
				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()

				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == labels.data)

			epoch_loss = (running_loss*1.0) / dataset_sizes[phase]
			epoch_acc = (running_corrects*1.0) / dataset_sizes[phase]

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			f=open("results.txt", "a+")
			f.write("epoch number - ")
			f.write(str(epoch))
			f.write("\n")
			f.write(str(phase))
			f.write("loss - ")
			f.write(str(epoch_loss))
			f.write("\n")
			f.write("accuracy - ")
			f.write(str(epoch_acc))
			f.write("\n")
			f.close()
			
				
			# deep copy the model
			if phase == 'test' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
			print('Best val Acc: {:4f}'.format(best_acc))	
		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	torch.save(best_model_wts,'trained_wei.pth')
	return model


#######################################################



model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
					   num_epochs=11)



