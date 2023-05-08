
# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import copy
import numpy as np
from torch.nn.modules import BatchNorm1d
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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load the dataset and apply the transform
    dataset = datasets.ImageFolder(root="dataset/", transform=transform)

    # Define a data loader for the dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load a pretrained Food101 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Remove the last layer to access the embeddings
    model.eval()
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Identity()

    # Extract the embeddings
    embeddings = []
    i = 0
    for images, _ in loader:
        print(f"Working: {(i/len(dataset))*100}%")

        with torch.no_grad():
            features_transformed = images.to(device)
            features_extracted = model(features_transformed)
            print(f"Shape of features_extracted: {features_extracted.shape}")
            embeddings.append(features_extracted.cpu().numpy())
        i += 1
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Embeddings: {embeddings.shape}")    
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
    # for i in range(len(embeddings)):
    #     embeddings_norm = np.linalg.norm(embeddings[i])
    #     embeddings[i] /= embeddings_norm
    
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
    y = np.vstack(y)

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
    def __init__(self, fstHL, sndHL, thdHL, fthHL, fiftHL, sixtHL, sevtHL):
        """
        The constructor of the model.
        """
        super().__init__()
        self.lin1 = nn.Linear(2048, fstHL)
        self.bn1 = nn.BatchNorm1d(fstHL)
        self.dropout1 = nn.Dropout(p=0.75)
        self.lin2 = nn.Linear(fstHL, sndHL)
        self.bn2 = nn.BatchNorm1d(sndHL)
        self.dropout2 = nn.Dropout(p=0.15)
        self.lin3 = nn.Linear(sndHL, thdHL)
        self.bn3 = nn.BatchNorm1d(thdHL)
        self.dropout3 = nn.Dropout(p=0.15)
        self.lin4 = nn.Linear(thdHL, fthHL)
        self.bn4 = nn.BatchNorm1d(fthHL)
        self.dropout4 = nn.Dropout(0.15)
        self.lin5 = nn.Linear(fthHL, fiftHL)
        self.bn5 = nn.BatchNorm1d(fiftHL)
        self.dropout5 = nn.Dropout(0.15)
        self.lin6 = nn.Linear(fiftHL, sixtHL)
        self.bn6 = nn.BatchNorm1d(sixtHL)
        self.dropout6 = nn.Dropout(0.15)
        self.lin7 = nn.Linear(sixtHL, sevtHL)
        self.bn7 = nn.BatchNorm1d(sevtHL)
        self.dropout7 = nn.Dropout(0.15)
        self.lin8 = nn.Linear(sevtHL, 128)


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        
        x = F.celu(self.bn1(self.lin1(x)), alpha=1)
        x = self.dropout1(x)
        x = F.celu(self.bn2(self.lin2(x)), alpha=1)
        x = self.dropout2(x) 
        x = F.celu(self.bn3(self.lin3(x)), alpha=1)
        x = self.dropout3(x)
        x = F.celu(self.bn4(self.lin4(x)), alpha=1)
        x = self.dropout4(x)
        x = F.celu(self.bn5(self.lin5(x)), alpha=1)
        x = self.dropout5(x)
        x = F.celu(self.bn6(self.lin6(x)), alpha=1)
        x = self.dropout6(x)
        x = F.celu(self.bn7(self.lin7(x)), alpha=1)
        x = self.dropout7(x)
        x = self.lin8(x)
        return x
        
    

def test_outputs(train_loader, model):
    model.eval()
    predictions = []
    values = []
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=1, shuffle=True)
        # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch, y] in train_loader:
            values.append(y)
            x_batch= x_batch.to(device)
            predicted1 = model(x_batch[:,:2048])
            predicted2 = model(x_batch[:,2048:2048*2])
            predicted3 = model(x_batch[:,2048*2:2048*3])
            predicted1 = predicted1.cpu().numpy()
            predicted2 = predicted2.cpu().numpy()
            predicted3 = predicted3.cpu().numpy()

            for j in range(len(predicted1)):

                norm1 = np.linalg.norm(predicted1[j] - predicted2[j])
                norm2 = np.linalg.norm(predicted1[j] - predicted3[j])
                # Rounding the predictions to 0 or 1
                if norm1 <= norm2:
                    predictions.append(1)
                else:
                    predictions.append(0)

        correct = 0
        for k in range(len(predictions)):
            if predictions[k] == values[k]:
                correct +=1
            
        print(f"Score: {correct/len(predictions)}%")


def train_model(train_loader, fstHL, sndHL, thdHL, fthHL, fiftHL, sixtHL, sevtHL, learning_rate):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    # Define the model architecture
    model = Net(fstHL, sndHL, thdHL, fthHL, fiftHL, sixtHL, sevtHL)

    # Set the model to train mode and move it to the device
    model.train()
    model.to(device) 

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
   
    critization = F.triplet_margin_loss
    
    # Define the number of epochs and the validation split
    n_epochs = 10
    validation_split = 0.2
    
    # Calculate the size of the validation set
    n_train_examples = len(train_loader.dataset)
    n_valid_examples = int(n_train_examples * validation_split)
    
    # Split the training data into training and validation sets
    train2_data, valid_data = torch.utils.data.random_split(train_loader.dataset, 
                                                           [n_train_examples - n_valid_examples, 
                                                            n_valid_examples])
    
    # Create data loaders for the training and validation sets
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    #train2_loader = torch.utils.data.DataLoader(train2_data, batch_size=batch_size, shuffle=True)
    #valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        # Train the model
    best_valid_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        k = 1
        
        # Train the model on the training data
        for [X, y] in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output1 = model.forward(X[:,:2048])
            output2 = model.forward(X[:,2048:2048*2])
            output3 = model.forward(X[:,2048*2:2048*3])

            batch_loss = torch.zeros(1, dtype=X.dtype, device=device)  # initialize batch loss to zero
            for i in range(X.size(0)):  # loop over rows in the batch
                if y[i] == 1:
                    loss = critization(output1[i], output2[i], output3[i])
                else:
                    loss = critization(output1[i], output3[i], output2[i])
                batch_loss += loss  # add loss for this row to batch loss

            batch_loss /= X.size(0)  # compute average loss per sample in the batch
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item() * X.size(0)

            if(k % 30 == 0):
                print(f"Epoch {epoch + 1} Training, Progress: {(k / len(train_loader)) * 100}%")
                #print(f"Current loss: {batch_loss.item()}, current X.size(1): {X.size(1)}")
                pass
            k +=1
        
        '''
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
                
                batch_loss = torch.zeros(1, dtype=X.dtype, device=device) 
                for i in range(X.size(0)):  # loop over rows in the batch
                    if y[i] == 1:
                        loss = critization(output1[i], output2[i], output3[i])
                    else:
                        loss = critization(output1[i], output3[i], output2[i])
                    batch_loss += loss 
                    batch_loss /= X.size(0)  
                    valid_loss += batch_loss.item() * X.size(0)
                   
        '''
        # Print the validation loss for this epoch
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch + 1} Train-Loss {train_loss}")
        '''
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
    for epoch in range(n_epochs):
        for [X, y] in train2_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output1 = model.forward(X[:,:2048])
            output2 = model.forward(X[:,2048:2048*2])
            output3 = model.forward(X[:,2048*2:2048*3])
            batch_loss = torch.zeros(1, dtype=X.dtype, device=device) 
            for i in range(X.size(0)):  # loop over rows in the batch
                if y[i] == 1:
                    loss = critization(output1[i], output2[i], output3[i])
                else:
                    loss = critization(output1[i], output3[i], output2[i])
                batch_loss += loss
                batch_loss /= X.size(0) 
                
                
                       
                
            batch_loss.backward()
            optimizer.step()

        '''
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
            predicted1 = model(x_batch[:,:2048])
            predicted2 = model(x_batch[:,2048:2048*2])
            predicted3 = model(x_batch[:,2048*2:2048*3])
            predicted1 = predicted1.cpu().numpy()
            predicted2 = predicted2.cpu().numpy()
            predicted3 = predicted3.cpu().numpy()


            norm1 = np.linalg.norm(predicted1 - predicted2)
            norm2 = np.linalg.norm(predicted1 - predicted3)
            # Rounding the predictions to 0 or 1
            if norm1 <= norm2:
                predictions.append(1)
            else:
                predictions.append(0)
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

    fstHL = 1024
    sndHL = 512
    thdHL = 512
    fthHL = 256
    fiftHL =256
    sixtHL = 128
    sevtHL = 128


    best_value = float('inf')
    
    
    best_model = train_model(train_loader, fstHL, sndHL, thdHL, fthHL, fiftHL, sixtHL, sevtHL, 0.001)
    test_outputs(train_loader, best_model)
   

    test_loader = create_loader_from_np(X_test, train = False, batch_size=1, shuffle=False)
    # test the model on the test data
    test_model(best_model, test_loader)
    print("Results saved to results.txt")


