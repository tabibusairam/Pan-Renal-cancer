import os
from shutil import copyfile

path = '/Neutron3/Datasets/med_img/valid/above8/'
path1 = '/Neutron3/Datasets/med_img/valid1/above8/'
for i in range(26):  #above8
	for j in range(2000):
		k=j+1
		l=i+1
		img = path+str(l)+'-'+str(k)+'.png'
		img1 = path1+str(l)+'-'+str(k)+'.png'
		print(img)
		#os.remove(img)
		copyfile(img,img1)

path = '/Neutron3/Datasets/med_img/valid/below7/'
path1 = '/Neutron3/Datasets/med_img/valid1/below7/'
for i in range(21):  #below7
	for j in range(2000):
		k=j+1
		l=i+1
		img = path+str(l)+'-'+str(k)+'.png'
		img1 = path1+str(l)+'-'+str(k)+'.png'
		print(img)
		#os.remove(img)
		copyfile(img,img1)
		






	


