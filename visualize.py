import preprocessing 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import params

from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid

#@title Default title text
## imshow works when we have transformed the image using some transformation
def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0)) ## This line of code corrects the dimension issue that occurs during transformation 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    axis.imshow(inp)

print(params.epoch)
print(params.study_type)
transformed_train_dataset,transformed_valid_dataset = preprocessing.preprocess()
transformed_train_dl = DataLoader(transformed_train_dataset,batch_size=params.batch_size,shuffle=True)
transformed_valid_dl = DataLoader(transformed_valid_dataset,batch_size=params.batch_size,shuffle=True)
imgview, label,Id = next(iter(transformed_valid_dl))
#print(img, label.size())
fig = plt.figure(1, figsize=(16, 8))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 8), axes_pad=0.05)  
for i in range(imgview.size()[0]):
  #print(img)
  ax = grid[i]
  print(imgview.shape)
  imshow(ax,imgview[i])
  
