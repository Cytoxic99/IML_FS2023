import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold


###############################################################################
# READ ME

# This code was written by Levi Lingsch, Fabiano Sasselli, Jonas Gruetter
# for Intro. to Machine Learning at ETH Zurich, Spring 2022.

# All data should be contained in .csv files within a subdirectory titled 'csv'

# A subdirectory title 'models' should exist for saving/loading models
# Pretrained models can be selected by setting the train_pre and train to 'False'

# The predictions are saved in the current directory in a file names 'outputs.csv'
###############################################################################
# autoencoder class
class HLG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.encoder2 = nn.Linear(
            in_features=256, out_features=128
        )
        self.encoder3 = nn.Linear(
            in_features=128, out_features=64
        )
        self.encoder4 = nn.Linear(
            in_features=64, out_features=32
        )
        self.encoder5 = nn.Linear(
            in_features=32, out_features=32
        )
        self.output_layer = nn.Linear(
            in_features=32, out_features=1
        )

    def forward(self, features):
        activation = self.encoder1(features)
        activation = torch.tanh(activation)
        code1 = self.encoder2(activation)
        code1 = torch.tanh(code1)
        code2 = self.encoder3(code1)
        code2 = torch.tanh(code2)
        code3 = self.encoder4(code2)
        code3 = torch.tanh(code3)
        code4 = self.encoder5(code3)
        code4 = torch.tanh(code4)
        output = self.output_layer(code4)
        return output


###############################################################################
# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}.')
model = HLG(input_shape=1000).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
###############################################################################
# train on LUMO data

# import pretrain data: 50,000 data samples, 1000 features
pre_data = pd.read_csv('./csv/pretrain_features.csv', index_col=0)
pre_data = pre_data.drop('smiles', axis=1).to_numpy()
pre_lbls = pd.read_csv('./csv/pretrain_labels.csv', index_col=0)
pre_lbls = pre_lbls.to_numpy()

# make a dataloader
data_tensor = torch.Tensor(pre_data)
lbls_tensor = torch.Tensor(pre_lbls)
pre_dataset = TensorDataset(data_tensor, lbls_tensor)
pre_dataloader = DataLoader(pre_dataset, batch_size=32)

train_pre = True  # if you want to train (True) or use saved model (False)
epochs = 8
model_path = f'./models/lumo_predictor_epochs-{epochs}.pth'
if train_pre:
    print('training model.')
    for epoch in range(epochs):
        loss = 0
        for batch_features, batch_labels in pre_dataloader:
            # load features
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            # zero optimizer
            optimizer.zero_grad()
            # calculate outputs
            outputs = model(batch_features)
            # calculate loss
            train_loss = criterion(outputs, batch_labels)
            # backpropogate
            train_loss.backward()
            # backpropogate
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(pre_dataloader)
        print(f'epoch: {epoch}, loss: {loss}')
    torch.save(model.state_dict(), model_path)
    print('model successfully saved.')
else:
    print('loading model.')
    model.load_state_dict(torch.load(model_path))

###############################################################################
# train on HLG data

# import train data: 100 data samples, 1000 features
train_data = pd.read_csv('./csv/train_features.csv', index_col=0)
train_data = train_data.drop('smiles', axis=1).to_numpy()
train_lbls = pd.read_csv('./csv/train_labels.csv', index_col=0)
train_lbls = train_lbls.to_numpy()

# make a dataloader
data_tensor = torch.Tensor(train_data)
lbls_tensor = torch.Tensor(train_lbls)
train_dataset = TensorDataset(data_tensor, lbls_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32)

# train the model additionally on the hlg data
epochs = 140
train_post = True
model_path = f'./models/hlg_predictor_epochs-{epochs}.pth'

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
if train_post:
    for epoch in range(epochs):
        loss = 0
        for batch_features, batch_labels in train_dataloader:
            # load features
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            # zero optimizer
            optimizer.zero_grad()
            # calculate outputs
            outputs = model(batch_features)
            # calculate loss
            train_loss = criterion(outputs, batch_labels)
            # backpropogate
            train_loss.backward()
            # backpropogate
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(pre_dataloader)
        print(f'epoch: {epoch}, loss: {loss}')
    torch.save(model.state_dict(), model_path)
    print('model successfully saved.')
else:
    print('loading model.')
    model.load_state_dict(torch.load(model_path))

###############################################################################
# predict on test features

# import test data: 10,000 data samples, 1000 features
test_data = pd.read_csv('./csv/test_features.csv', index_col=0)
test_data = test_data.drop('smiles', axis=1).to_numpy()
sample = pd.read_csv('./csv/sample.csv', index_col=0)

# make a dataloader
test_tensor = torch.Tensor(test_data)
sample_tensor = torch.Tensor(sample.to_numpy())
test_dataset = TensorDataset(test_tensor, sample_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=1)

pred = torch.zeros(sample_tensor.shape)
with torch.no_grad():
    for index, (batch_features, _) in enumerate(test_dataloader):
        # load features
        batch_features = batch_features.to(device)
        # calculate outputs
        outputs = model(batch_features)
        pred[index] = outputs
test_predictions = sample
test_predictions.loc[:, 'y'] = pred.numpy()
test_predictions.to_csv('outputs.csv')