# Another tedious job in computer vision task - need to preprocess the data and load in data loader - Preprocessing

import PIL
print(PIL.PILLOW_VERSION)
import torch 
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
from torchvision import transforms, utils
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, transform
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd 

def preprocess()


	train_set = pd.read_csv('MURA-v1.1/train_image_paths.csv',header=None)
	valid_set = pd.read_csv('MURA-v1.1/valid_image_paths.csv',header=None)
	train_set.columns = ['Path']
	valid_set.columns = ['Path']

	def study_label_create(df) : 
	  df['label'] = 0 
	  df.loc[(df.Path.apply(lambda x: x.find('positive')) > 0 ),['label']] = 1
	  df['Unique_Id'] = df.index
	  df['Study_type'] = ''
	  df.loc[(df.Path.apply(lambda x: x.find('XR_SHOULDER')) > 0 ),['Study_type']] = 'XR_SHOULDER'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_FINGER')) > 0 ),['Study_type']] = 'XR_FINGER'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_FOREARM')) > 0 ),['Study_type']] = 'XR_FOREARM'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_HAND')) > 0 ),['Study_type']] = 'XR_HAND'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_HUMERUS')) > 0 ),['Study_type']] = 'XR_HUMERUS'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_WRIST')) > 0 ),['Study_type']] = 'XR_WRIST'
	  df.loc[(df.Path.apply(lambda x: x.find('XR_ELBOW')) > 0 ),['Study_type']] = 'XR_ELBOW'

	  return df

	train_set = study_label_create(train_set)
	valid_set = study_label_create(valid_set)

	train_path_df = train_set[train_set.Study_type == 'params.study_type']
	valid_path_df = valid_set[valid_set.Study_type == 'params.study_type']

	class MuraImageDataset(Dataset):
		"""Mura dataset."""
		def __init__(self, df, root_dir, transform=None):
			"""
			Args:
				df (dataframe): Path to the image file with annotations.
				root_dir (string): Directory with all the images.
				transform (callable, optional): Optional transform to be applied
					on a sample.
			"""
			self.df = df
			self.root_dir = root_dir
			self.transform = transform
		def __len__(self):
			return len(self.df)
		def __getitem__(self, idx):
			img_name = os.path.join(self.root_dir,
									self.df.iloc[idx, 0])
					
			### -- Try Clahe Transformation -- ##
			img = cv2.imread(img_name,0)
			#create a CLAHE object (Arguments are optional).
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			img = clahe.apply(img)
			image = np.stack((img,)*3, -1)
			### -- Clahe Transformation Ends -- ##
			Id = self.df.iloc[idx, 2]
			labels = self.df.iloc[idx, 1]
			labels = labels.astype('float')
			if self.transform:
				image = self.transform(image)
			return [image, labels,Id]
			
	transformed_train_dataset = MuraImageDataset(df=train_path_df,
										root_dir='/content',
										transform=transforms.Compose([
												   transforms.ToPILImage(),
												   transforms.RandomRotation(10),
												   #transforms.Resize(224),
												   transforms.CenterCrop(224),
												   transforms.RandomHorizontalFlip(),
												   transforms.ToTensor(),
												   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
												   
											   ]))

	transformed_valid_dataset = MuraImageDataset(df=valid_path_df,
										root_dir='/content',
										transform=transforms.Compose([
												   transforms.ToPILImage(),
												   transforms.RandomRotation(10),
												   #transforms.Resize(224),
												   transforms.CenterCrop(224),
												   transforms.RandomHorizontalFlip(),
												   transforms.ToTensor(),
												   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
												   
											   ]))

	transformed_train_dl = DataLoader(transformed_train_dataset,batch_size=params.batch_size,shuffle=True)
	transformed_valid_dl = DataLoader(transformed_valid_dataset,batch_size=params.batch_size,shuffle=True)
	
return transformed_train_dl,transformed_valid_dl