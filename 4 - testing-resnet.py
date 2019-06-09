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



###########################################################################################


model_conv = torchvision.models.resnet18(pretrained=True)
#print(model_conv)
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

print(model_conv)
#print('-'*50)
#params = list(model_conv.parameters())
#print(params)
if use_gpu:
	model_conv = (model_conv).cuda()
model_conv.load_state_dict(torch.load('/shared/supernova/home/sairam.tabibu/results/kidney/subtype/resnet-18-subtype_classification/resnet_wei1.pth'))


############################################################################################

#data_dir = '/tmp/kidney/'
data_dir = '/shared/supernova/home/sairam.tabibu/kidney-normal/'
#data_dir = '/Users/user/Desktop/medical_images'
phase = 'test'
resize = [224,224]
data_transforms = {
        'test': transforms.Compose([
            #transforms.TenCrop(max(resize)),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            #transforms.Lambda(lambda crops: torch.stack([transforms.Normalize([0.620, 0.446, 0.594], [0.218, 0.248, 0.193])(crop) for crop in crops])),
            transforms.Resize(max(resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.596, 0.436, 0.586], [0.2066, 0.240, 0.186])			        
        ]),
    }


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
											 shuffle=False)
			  for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
class_names = image_datasets['test'].classes




###################################################t



def train_model(model, num_epochs):
	since = time.time()


	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		running_corrects = 0
		running_corr = 0
		# Each epoch has a training and val idation phase
		for phase in ['test']:
			print(phase)

			model.train(False)	  
			running_loss = 0.0
			running_corrects = 0
			k=1;
			# Iterate over data.
			for data in dataloaders[phase]:
				k=k+1;
				if(k%100 == 0):
					print(k/100)
					
				# get the inputs
				inputs, labels, idx = data

				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs = Variable(inputs)
					labels = Variable(labels)

				# zero the parameter gradients

				#bs, ncrops, c, h, w = inputs.size()
				#inp_chg = inputs.view(-1, c, h, w)
				outputs = model(inputs)	
				#outputs = F.tanh(outputs)
				#outputs = (outputs+1)/2
					
				#outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
				#print(outputs)
				_, preds = torch.max(outputs.data, 1)
				f = open("/shared/supernova/home/sairam.tabibu/probabilities.txt","a+")
				f.write(str(torch.max(outputs).item()) + "\n" )
				f.close()

				f = open("/shared/supernova/home/sairam.tabibu/classes.txt","a+")
				f.write(str(preds.item()) + "\n" )
				f.close()
			
				running_corrects += torch.sum(preds == labels.data)
					
				#print('running loss',running_loss)
			epoch_acc = (running_corrects.double()) / dataset_sizes[phase]

			print('{} resnet-18-Acc: {:.4f}'.format(
				phase, epoch_acc))
			print(running_corrects)
			print(dataset_sizes[phase])


		print()

	time_elapsed = time.time() - since
	print('testing complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))


	return model


#######################################################



model_ft = train_model(model_conv, num_epochs=1)



