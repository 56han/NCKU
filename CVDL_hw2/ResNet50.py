import torch
import torchvision
from torchvision import datasets, transforms
# import tensorflow as tf
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torchsummary import summary
from tqdm import tqdm
# Replace the last fully connected layer with a new one for binary classification
import torch.nn as nn

traindir = r".\Dataset_Cvdl_Hw2_Q5\dataset\training_dataset"
testdir = r".\Dataset_Cvdl_Hw2_Q5\dataset\validation_dataset"

#transformations
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  
    transforms.RandomHorizontalFlip(),
    # transforms.RandomErasing(),
    torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],),
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],),
])

#datasets
train_data = datasets.ImageFolder(traindir, transform = train_transforms)
test_data = datasets.ImageFolder(testdir, transform = test_transforms)

#dataloader
trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size = 64)

def make_train_step(model, optimizer, loss_fn):
    def train_step(x,y):
        #make prediction
        yhat = model(x)
        #enter train mode
        model.train()
        #compute loss
        loss = loss_fn(yhat,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #optimizer.cleargrads()

        return loss
    return train_step

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the pre-trained ResNet50 model
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze the parameters of the pre-trained model
for param in resnet.parameters():
    param.requires_grad = False

# Replace the output layer
num_features = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)

# Define the loss function and optimizer
loss_fn = BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)

#train step
train_step = make_train_step(resnet, optimizer, loss_fn)

# Print the model architecture

resnet = resnet.to(device)
# summary(resnet, (3, 224, 224))


losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []

n_epochs = 40
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

for epoch in range(n_epochs):
    epoch_loss = 0
    for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate ove batches
        x_batch , y_batch = data
        x_batch = x_batch.to(device) #move to gpu
        y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
        y_batch = y_batch.to(device) #move to gpu


        loss = train_step(x_batch, y_batch)
        epoch_loss += loss/len(trainloader)
        losses.append(loss)
    
    epoch_train_losses.append(epoch_loss)
    print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

    #validation (doesnt requires gradient)
    with torch.no_grad():
        cum_loss = 0
        correct_predictions = 0
        total_samples = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device)

            #model to eval mode
            resnet.eval()

            yhat = resnet(x_batch)
            val_loss = loss_fn(yhat,y_batch)
            cum_loss += loss/len(testloader)
            val_losses.append(val_loss.item())
            # Compute accuracy
            predicted_labels = (yhat >= 0.5).float()  # threshold of 0.5 for binary classification
            correct_predictions += (predicted_labels == y_batch).sum().item()
            total_samples += y_batch.size(0)

        epoch_test_losses.append(cum_loss)
        accuracy = correct_predictions / total_samples
        print('Epoch : {}, Val Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch + 1, cum_loss, accuracy))  
        
        best_loss = min(epoch_test_losses)
        
        #save best model
        if cum_loss <= best_loss:
            best_model_wts = resnet.state_dict()
        
        #early stopping
        early_stopping_counter = 0
        if cum_loss > best_loss:
            early_stopping_counter +=1

        if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
            print("/nTerminating: early stopping")
            break #terminate training
    
#load best model
resnet.load_state_dict(best_model_wts)
torch.save(best_model_wts, r".\best_resnet50_model.pth")