import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torchvision.models as models

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('GPU state:', device)


#----------------------------------------------
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomResizedCrop(32), #scaling
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # 同樣的歸一化操作，將影像像素值對應到平均值為0，標準差為1的範圍。
])
#----------------------------------------------

batch_size = 8

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=val_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

 
classes = ('airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class MyVGG19_BN(nn.Module):
    def __init__(self, num_classes=10, dropout: float = 0.5):
        super(MyVGG19_BN, self).__init__()
        pretrain_model = models.vgg19_bn(pretrained = True)
        pretrain_model.classifier = nn.Sequential()  # remove last layer
        self.features = pretrain_model
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = MyVGG19_BN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


train_accuracy_list = []
train_loss_list = []
valid_accuracy_list = []
valid_loss_list = []

validloader = testloader
num_epochs = 40

for epoch in range(num_epochs):  # loop over the dataset multiple times
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    train_loss_temp = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        train_loss_temp += loss.item()
        predict = torch.argmax(outputs, 1)
        train_correct += (predict == labels).sum().item()
        train_total += labels.size(0)
        train_accuracy = train_correct / train_total
        del inputs, labels
        torch.cuda.empty_cache()

        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[Train | {epoch + 1}, {i + 1:5d}] train_loss: {train_loss_temp / 2000:.5f}, train_acc = {train_accuracy: .5f}')
            print(f'{train_loss_temp / 2000:.5f}')
            train_loss_temp = 0.0

    train_accuracy = train_correct / train_total
    train_loss = train_loss / len(trainloader) 
    print(f'[Train | {epoch + 1:03d}, {num_epochs:3d}] train_loss: {train_loss:.5f}, train_acc = {train_accuracy:.5f}')

    train_accuracy_list.append(int((train_accuracy * 100) + 0.5))
    train_loss_list.append(round(train_loss, 2))

    model.eval()
    valid_correct = 0
    valid_total = 0
    valid_loss = 0.0
    valid_loss_temp = 0.0
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward + backward + optimize
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        valid_loss += loss.item()
        valid_loss_temp += loss.item()
        predict = torch.max(outputs.data, 1)[1]
        valid_correct += (predict == labels).sum().item()
        valid_total += labels.size(0)
        del inputs, labels
        torch.cuda.empty_cache()
        valid_accuracy = valid_correct / valid_total
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[Valid | {epoch + 1}, {i + 1:5d}] valid_loss: {valid_loss_temp / 2000:.5f}, valid_acc = {valid_accuracy: .5f}')
            print(f'{valid_loss_temp / 2000:.5f}')
            valid_loss_temp = 0.0

    valid_loss = valid_loss / len(validloader)
    valid_accuracy = valid_correct / valid_total
    print(f'[Valid | {epoch + 1:03d}, {num_epochs:3d}] valid_loss: {valid_loss:.5f}, valid_acc = {valid_accuracy:.5f}')
    valid_accuracy_list.append(int((valid_accuracy * 100)+ 0.5))
    valid_loss_list.append(round(valid_loss, 2))
PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)
print('Finished Training')

print("train_accuracy_list:", train_accuracy_list)
print("train_loss_list:", train_loss_list)
print("valid_accuracy_list:", valid_accuracy_list)
print("valid_loss_list:", valid_loss_list)

x = [i for i in range(1, num_epochs + 1)]
plt.plot(x,train_loss_list,color="blue",label="Train Loss")
plt.plot(x,valid_loss_list,color="orange",label="Validation Loss")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend(loc = 'upper right')
plt.title("Loss")
plt.savefig('loss.png')
plt.show()

x = [i for i in range(1, num_epochs + 1)]
plt.plot(x,train_accuracy_list,color="blue",label="Train Accuracy")
plt.plot(x,valid_accuracy_list,color="orange",label="Validation Accuracy")
plt.xlabel("Epoch Number")
plt.ylabel("Accuracy(%)")
plt.legend(loc = 'upper left')
plt.title("Accuracy")
plt.savefig('Accuracy.png')
plt.show() 