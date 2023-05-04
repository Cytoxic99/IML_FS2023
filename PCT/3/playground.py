import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

def get_data(file, train=True):
 """
 Load the triplets from the file and generate the features and labels.

 input: file: string, the path to the file containing the triplets
       train: boolean, whether the data is for training or testing

 output: X: numpy array, the features
         y: numpy array, the labels
 """
 triplets = []
 with open(file) as f:
  for line in f:
   triplets.append(line)

 # generate training data from triplets
 train_dataset = datasets.ImageFolder(root="dataset/",
                                      transform=None)
 filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
 embeddings = np.load('dataset/embeddings.npy')

 # TODO: Normalize the embeddings across the dataset
 file_to_embedding = {}

 for i in range(len(filenames)):
  file_to_embedding[filenames[i].replace("food\\", "")] = embeddings[i]
 print(file_to_embedding)
 X = []
 y = []

 # use the individual embeddings to generate the features and labels for triplets
 for t in triplets:
  emb = [file_to_embedding[a] for a in t.split()]
  X.append(np.hstack([emb[0], emb[1], emb[2]]))
  y.append(1)
  # Generating negative samples (data augmentation)
  if train:
   X.append(np.hstack([emb[0], emb[2], emb[1]]))
   y.append(0)
 X = np.vstack(X)
 y = np.hstack(y)
 return X, y

X_train, y_train = get_data('train_triplets.txt')
#print(X_train)