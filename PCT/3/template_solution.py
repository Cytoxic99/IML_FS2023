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
batch_size = 256

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # Define a transform to pre-process the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset and apply the transform
    dataset = datasets.ImageFolder(root="dataset/", transform=transform)

    # Define a data loader for the dataset
    loader = DataLoader(dataset=dataset,
                        batch_size=64,
                        shuffle=False,
                        pin_memory=True, num_workers=16)

    # Load a pretrained ResNet model
    model = resnet50(weights = ResNet50_Weights.DEFAULT)
    # Remove the last layer to access the embeddings
    model.eval()
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Identity()

    # Extract the embeddings
    embeddings = []
    i = 1
    for images, _ in loader:
        print(f"Working: {(i/len(loader))*100}%")
        images = images.to(device)
        with torch.no_grad():
            features = model(images)
        embeddings.append(features.cpu().numpy().reshape(images.shape[0], -1))
        i += 1
    embeddings = np.concatenate(embeddings, axis=0)


    # for images, _ in loader:
    #     with torch.no_grad():
    #         features = model(images)
    #     embeddings.append(features.cpu().numpy().reshape(images.shape[0], -1))
    # embeddings = np.concatenate(embeddings, axis=0)

    # Save the embeddings to a file
    np.save('dataset/embeddings.npy', embeddings)





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
    print(f"Size of embeddings: {embeddings.shape}")
    for i in range(len(embeddings)):
        embeddings_norm = np.linalg.norm(embeddings[i])
        embeddings[i] /= embeddings_norm
    
    # use the individual embeddings to generate the features and labels for triplets
    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
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
    def __init__(self, fstHL, sndHL,thdHL):
        """
        The constructor of the model.
        """
        super().__init__()
        self.lin1 = nn.Linear(2048, fstHL)
        self.lin2 = nn.Linear(fstHL, sndHL)
        self.lin3 = nn.Linear(sndHL, thdHL)
        self.lin4 = nn.Linear(thdHL, 128)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.sigmoid(self.lin4(x))
        return x
        
    

def train_model(train_loader, fstHL, sndHL, thdHL, learning_rate):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    # Define the model architecture
    model = Net(fstHL, sndHL, thdHL)

    # Set the model to train mode and move it to the device
    model.train()
    model.to(device) 

    
   
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
    critization = nn.TripletMarginLoss(margin=1.0)
    
    # Define the number of epochs and the validation split
    n_epochs = 10
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
        i = 1
        
        # Train the model on the training data
        for [X, y] in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output1 = model.forward(X[:,:2048])
            output2 = model.forward(X[:,2048:2048*2])
            output3 = model.forward(X[:,2048*2:2048*3])

            
            if y[0] == 1:
                loss = critization(output1, output2, output3)
            else:
                loss = critization(output1, output3, output2)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)

            if(i % 30 == 0):
                #print(f"Epoch {epoch + 1} Training, Progress: {(i / len(train_loader)) * 100}%")
                #print(f"Current loss: {loss.item()}, current X.size(1): {X.size(1)}")
                print(f"Current Trainloss: {abs(loss.item() - 1)}")
                
            i += 1
            
        # Evaluate the model on the validation data
        model.eval()
        with torch.no_grad():
            i = 0
            for [X, y] in valid_loader:
                X = X.to(device)
                y = y.to(device)
                output1 = model.forward(X[:,:2048])
                output2 = model.forward(X[:,2048:2048*2])
                output3 = model.forward(X[:,2048*2:2048*3])
                if y[0] == 1:
                    loss = critization(output1, output2, output3)
                else:
                    loss = critization(output1, output3, output2)
                
                valid_loss += loss.item() * X.size(0)
                if(i % 1000 == 0):
                    #print(f"Epoch {epoch + 1} Training, Progress: {(i / len(valid_loader)) * 100}%")
                    #print(f"Current loss: {loss.item()}")
                    pass
                i += 1
        
        # Print the validation loss for this epoch
        train_loss /= len(train_data)
        valid_loss /= len(valid_data)
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))
        
        # Save the best model based on the validation loss evtl löschen
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict()
        
    # Train the best model on the whole training data
    model.load_state_dict(best_model)
    model.train()
    #print(f"Best validation loss: {best_valid_loss}")
    for [X, y] in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output1 = model.forward(X[:,:2048])
        output2 = model.forward(X[:,2048:2048*2])
        output3 = model.forward(X[:,2048*2:2048*3])
        if y[1] == 1:
            loss = critization(output1, output2, output3)
        else:
            loss = critization(output1, output3, output2)
            
        loss.backward()
        optimizer.step()

        
    return model, best_valid_loss


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
            print(f"Size x_batch: {x_batch.shape}")
            predicted1 = model(x_batch[:,:2048])
            predicted2 = model(x_batch[:,2048:2048*2])
            predicted3 = model(x_batch[:,2048*2:2048*3])
            predicted1 = predicted1.cpu().numpy()
            predicted2 = predicted2.cpu().numpy()
            predicted3 = predicted3.cpu().numpy()


            norm1 = np.linalg.norm(predicted1 - predicted2)
            norm2 = np.linalg.norm(predicted1 - predicted3)
            # Rounding the predictions to 0 or 1
            if norm1 >= norm2:
                predictions.append(0)
            else:
                predictions.append(1)
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
    X_test, _ = get_data(TEST_TRIPLETS, train=False)


    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)

    fstHL = 256
    sndHL = 128
    thdHL = 128
    


    best_value = float('inf')
    
    
    best_model, value = train_model(train_loader, fstHL, sndHL, thdHL, 0.001)

   

    test_loader = create_loader_from_np(X_test, train = False, batch_size=1, shuffle=False)
    # test the model on the test data
    test_model(best_model, test_loader)
    print("Results saved to results.txt")


