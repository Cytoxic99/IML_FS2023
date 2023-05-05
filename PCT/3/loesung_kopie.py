# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import copy
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim



if torch.cuda.is_available():
    device = torch.device('cuda')
    print("I have the GPU")
else:
    device = torch.device('cpu')
    print("Have to use CPU")

num_workers = 12
batch_size = 1
def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, num_workers=num_workers)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

     # Generate embeddings for all images in the dataset
    embeddings = []
    i = 0
    for images, labels in train_loader:
        i += 1.0
        print(f"working: {i/len(train_loader)}%")
        with torch.no_grad():
            # Forward pass through the model to extract the embeddings
            outputs = model(images.to(device))
            embeddings.append(outputs)

    # Concatenate the embeddings into a single numpy array
    embeddings = torch.cat(embeddings).numpy()

    # Save the embeddings to a file
    np.save('dataset/embeddings.npy', embeddings)
    
    return embeddings


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
    b = embeddings.shape
    # TODO: Normalize the embeddings across the dataset

    file_to_embedding = {}

    a = len(filenames)
    for i in range(len(filenames)):
        file_to_embedding[filenames[i].replace("food\\", "")] = embeddings[i]   #every of the 10k keys(image numbersa) gets an value(emdding row) of 1k numbers which is the i'th embeddings row.

    X = []
    y = []

    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        #c = np.array(emb).shape
        #emb = list of embeddings belonging to image numbers chosen by the values of a triplet row
        #shape of emb: 3x1000

        X.append(np.hstack([emb[0], emb[1], emb[2]])) #hängt einfach alle 3 embeddings hintereinander
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X) #shape = (2x #triplets) x 3000
    #c = np.array(X).shape
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=batch_size, shuffle=True, num_workers = num_workers):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.float))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.fc = nn.Linear(3000, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        x = self.fc(x)
        x = F.relu(x)
        return x
    
    

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    # Define the model architecture
    model = Net()
    
    # Set the model to train mode and move it to the device
    model.train()
    model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define the number of epochs and the validation split
    n_epochs = 1
    validation_split = 0.2
    
    # Calculate the size of the validation set
    n_train_examples = len(train_loader.dataset)
    n_valid_examples = int(n_train_examples * validation_split)
    
    # Split the training data into training and validation sets
    train_data, valid_data = torch.utils.data.random_split(train_loader.dataset, 
                                                           [n_train_examples - n_valid_examples, 
                                                            n_valid_examples])
    
    # Create data loaders for the training and validation sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    
    # Train the model
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        
        # Train the model on the training data
        for [X, y] in train_loader:
            optimizer.zero_grad()
            output = model.forward(X)
            y = y.unsqueeze(0)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            print(f"Epoch: {epoch + 1} Train loss: {loss.item()}")
            
        # Evaluate the model on the validation data
        model.eval()
        with torch.no_grad():
            for [X, y] in valid_loader:
                output = model(X)
                loss = criterion(output, y)
                valid_loss += loss.item() * X.size(0)
        
        # Print the validation loss for this epoch
        train_loss /= len(train_data)
        valid_loss /= len(valid_data)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))
        
        # Save the best model based on the validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()
    
    # Train the best model on the whole training data
    model.load_state_dict(best_model)
    model.train()
    for [X, y] in train_loader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
    return model


def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            print(f"The length of predicted is: {len(predicted)}")
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    #d = np.array(X).shape
    X_test, _ = get_data(TEST_TRIPLETS, train=False)



    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=1, shuffle=False)


    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
