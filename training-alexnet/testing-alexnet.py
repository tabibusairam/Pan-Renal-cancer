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


model_conv = torchvision.models.alexnet(pretrained=True)
#print(model_conv)
new_model_conv=(list(model_conv.classifier.children())[:-1])
new_model_conv.append(nn.Linear(4096, 2))
model_conv.classifier = nn.Sequential(*new_model_conv)

#model_conv.load_state_dict(torch.load('trained_wei.pth'))

print(model_conv)
if use_gpu:
	model_conv = nn.DataParallel(model_conv).cuda()

model_conv.load_state_dict(torch.load('trained_wei.pth'))

criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#################################################

data_dir = '/Neutron3/Datasets/med_img/'
#data_dir = '/Users/user/Desktop/medical_images1'

resize=[227,227];
data_transforms = {
	'valid': transforms.Compose([
		transforms.Resize(int(max(resize))),
		#transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
											 shuffle=False)
			  for x in ['valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['valid']}
class_names = image_datasets['valid'].classes




#####################################################



def train_model(model, criterion, optimizer, scheduler, num_epochs):
	since = time.time()

	best_model_wts = model.state_dict()
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		running_corrects = 0
		running_corr = 0
		# Each epoch has a training and val idation phase
		for phase in ['valid']:
			print(phase)

			model.train(False)	  
			running_loss = 0.0
			running_corrects = 0
			k=1;
			# Iterate over data.
			for data in dataloaders[phase]:

					
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
				outputs = model(inputs)
				#print(outputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)


				running_loss += loss.data[0]
				


				if(k==61):
					if(running_corrects > 1500):
						running_corr += 1
					running_corrects = 0
					k=1		

				if(k<=60):
					print(k)
					running_corrects += torch.sum(preds == labels.data)
					k=k+1;
					
				#print('running loss',running_loss)

			epoch_loss = (running_loss*1.0) / dataset_sizes[phase]
			epoch_acc = (running_corr*1.0) / (dataset_sizes[phase]/3000)

			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))


			# deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))


	return model


#######################################################



model_ft = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler,
					   num_epochs=1)



