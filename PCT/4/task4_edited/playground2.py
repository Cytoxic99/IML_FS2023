# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os


from sklearn.model_selection import train_test_split


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None

    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles",
                                                                                                  axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test


class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """

    def __init__(self, **kwargs):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data
        # and then used to extract features from the training and test data.

        # self.lin1 = nn.Linear(1000, 256)
        self.lin1 = nn.Linear(in_features=kwargs["input_shape"], out_features=256)
        nn.init.kaiming_uniform_(self.lin1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin2 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.lin2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin3 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.lin3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin4 = nn.Linear(64, 1)
        nn.init.kaiming_uniform_(self.lin4.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture
        # defined in the constructor.
        x = F.celu(self.lin1(x))
        x = F.celu(self.lin2(x))
        x = F.celu(self.lin3(x))
        # x = F.celu(self.lin4(x))
        x = self.lin4(x)
        return x


def pretraining_model(x, y, batch_size=256, eval_size=1000, lr=0.001, num_workers=12, shuffle=False, n_epoch=10):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set

    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    if os.path.exists('model_full.pth') == False:
        # Pretraining data loading
        in_features = x.shape[-1]
        x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
        x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
        y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
        print('pretraining data loaded')

        # TODO create data loaders
        train_dataset = TensorDataset(x_tr, y_tr)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True, num_workers=num_workers)

        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                pin_memory=True, num_workers=num_workers)

        print('pretraining loaders created')

        # model declaration
        model = Net(input_shape=1000)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_func = F.huber_loss

        # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set to monitor the loss.
        for epoch in range(n_epoch):
            # training
            train_loss = 0.0
            progress = 1
            for [x, y] in train_loader:
                optimizer.zero_grad()
                pred = model.forward(x).squeeze(1)
                loss = loss_func(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # print training progress
                # print(f'training epoch {epoch+1}: {progress/len(train_loader)*100} %')
                if progress % 30 == 0:
                    print(f"Epoch {epoch + 1} Training, Progress: {(progress / len(train_loader)) * 100}%")
                    pass
                progress += 1
            train_loss /= len(train_loader)

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for [inputs, targets] in val_loader:
                    outputs = model(inputs).squeeze(1)
                    loss = loss_func(outputs, targets)
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            print(f"Epoch: {epoch + 1}/{n_epoch}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")

        torch.save(model, 'pretrain_model_full.pth')
        print('pretraining model generated and saved')
    else:
        print('model is already trained and saved')


def training(x, y, batch_size=256, eval_size=10, lr=0.001, num_workers=12, shuffle=False, n_epoch=10):
    # Pretraining data loading
    # in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    print('training data loaded')

    # create dataloaders
    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=True, num_workers=num_workers)

    print('training loaders created')

    # model declaration
    model = torch.load('pretrain_model_full.pth')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.huber_loss

    # training loop
    for epoch in range(n_epoch):
        # training
        train_loss = 0.0
        progress = 1
        for [x, y] in train_loader:
            optimizer.zero_grad()
            pred = model.forward(x).squeeze(1)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # print training progress
            if (progress % 30 == 0):
                print(f"Epoch {epoch + 1} Training, Progress: {(progress / len(train_loader)) * 100}%")
                pass
            progress += 1
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for [inputs, targets] in val_loader:
                outputs = model(inputs).squeeze(1)
                loss = loss_func(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch: {epoch + 1}/{n_epoch}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")

        torch.save(model, 'train_model_full.pth')
        print('raining model generated and saved')


def test(x):
    x_test = torch.tensor(x, dtype=torch.float)
    model = torch.load('train_model_full.pth')
    model.eval()

    """
    pred = torch.zeros(x_test.shape)
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

    """

    with torch.no_grad():
        y_pred = model(x_test)

    y_pred = y_pred.numpy()

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")


#######################################################################################################################
# Main function. You don't have to change this

if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()

    # x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)

    print("data loaded")

    # generate pretrained model on the 5000 Lumo-labeled molecules and save it
    learning_rate = 0.0009
    n_eval = 1000
    batsch_size = 256
    num_arbeiter = 8
    epochen = 12
    mischen = False
    pretraining_model(x_pretrain, y_pretrain, batsch_size,
                      n_eval, learning_rate, num_arbeiter, mischen, epochen)

    # train the model on the 100 training data molecules
    learning_rate = 0.0009
    n_eval = 1000
    batsch_size = 256
    num_arbeiter = 8
    epochen = 12
    mischen = True
    training(x_pretrain, y_pretrain, batsch_size,
             n_eval, learning_rate, num_arbeiter, mischen, epochen)

    # test model and generate results file
    x_test = x_test.values
    test(x_test)
