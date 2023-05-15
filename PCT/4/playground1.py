# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



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
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.lin1 = nn.Linear(1000, 256)
        nn.init.kaiming_uniform_(self.lin1.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin2 = nn.Linear(256, 128)
        nn.init.kaiming_uniform_(self.lin2.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin3 = nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.lin3.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        self.lin4 = nn.Linear(64,1)
        nn.init.kaiming_uniform_(self.lin4.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')



    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = F.celu(self.lin1)
        x = F.celu(self.lin2)
        x = F.celu(self.lin3)
        x = F.celu(self.lin4)
        x = self.lin5
        return x


def make_feature_extractor(x, y, batch_size=256, eval_size=1000, lr=0.001, num_workers=12, shuffle=False, n_epoch=10):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    print('pretraining data loaded')

    #TODO create data loaders
    train_dataset = TensorDataset(x_tr, y_tr)
    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)

    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True, num_workers=num_workers)

    print('pretraining loaders created')

    # model declaration
    model = Net()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.huber_loss

    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set to monitor the loss.
    for epoch in range(n_epoch):
        # training
        train_loss = 0.0
        for [x, y] in train_loader:
            optimizer.zero_grad()
            pred = model.forward(x) #.squeeze(1)
            loss = loss_func(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for [inputs, targets] in val_loader:
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch: {epoch + 1}/{n_epoch}, training loss: {train_loss:.4f}, validation loss: {val_loss:.4f}")


    #evaluation
    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        return x

    return make_features


def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = None
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")

    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    learning_rate = 0.001
    n_eval = 1000
    batsch_size = 256
    num_arbeiter = 8
    epochen = 2
    mischen = True
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain, batsch_size,
                                                n_eval, learning_rate, num_arbeiter, mischen, epochen)
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    pipeline = Pipeline('pretrained', PretrainedFeatureClass, 'regressor', regression_model)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("results.csv", index_label="Id")
    print("Predictions saved, all done!")