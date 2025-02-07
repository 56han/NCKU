import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import copy
import pickle as pkl

DOANLOAD_DATASET = True
LR = 0.001
BATCH_SIZE = 16 # 一次epoch 32張
EPOCH = 40
MODELS_PATH = './models'
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(32, 4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomResizedCrop(32), #scaling
    torchvision.transforms.RandomVerticalFlip(),
    # torchvision.transforms.autoaugment.TrivialAugmentWide(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=mean, std=std)
    # torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # 這個操作將影像資料從範圍[0, 255]轉換為範圍[0, 1]的浮點張量，並重新排列通道的順序，通常是從 (H x W x C) 到 (C x H x W) 的轉換。
    torchvision.transforms.Normalize(mean=mean, std=std)
    # torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)) # 同樣的歸一化操作，將影像像素值對應到平均值為0，標準差為1的範圍。
])

train_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=True,
    transform=train_transform,
    download=DOANLOAD_DATASET
)

val_data = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=False,
    transform=val_transform,
    download=DOANLOAD_DATASET
)
# 這樣能在訓練時一次讀取1個Batch_size的數據而不用讀取整個Daset的數據
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = Data.DataLoader(dataset=val_data,batch_size=BATCH_SIZE,shuffle=False)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        pretrain_model = models.vgg19_bn(pretrained=True)
        pretrain_model.classifier = nn.Sequential()  # remove last layer
        self.features = pretrain_model

        # self.features = nn.Sequential(
        #     # 第一层
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # 第二层
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # 第三层
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # 第四层
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     # 第五层
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 10)  # 10 classes for classification
        )
    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
VGG19 = CustomVGG19()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VGG19 = VGG19.to(device)

# optimizer = torch.optim.Adam(VGG19.parameters(), lr=LR)
optimizer = torch.optim.SGD(VGG19.parameters(), lr=LR, momentum=0.9) #*****


loss_function = nn.CrossEntropyLoss()
training_acc = []
training_loss = []
val_acc = []
val_loss = []
valing_acc = []
valing_loss = []
best_acc = 0.0

hyperParameter = []
hyperParameter.append(BATCH_SIZE)
hyperParameter.append(LR)
hyperParameter.append(optimizer)

# train model
for epoch in range(EPOCH):
    print('\n', '*' * 25, 'epoch {}'.format(epoch + 1), '*' * 25)
    VGG19.train()
    correct = 0.0
    total = 0.0
    running_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        images = Variable(x, requires_grad=False)
        labels = Variable(y, requires_grad=False)
        # images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        out = VGG19(images) #b_x: images
        optimizer.zero_grad()
        loss = loss_function(out, labels) #b_y: labels
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        prob = torch.softmax(out, 1)
        _, predicted = torch.max(out, 1)
        # total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / len(train_data)
    train_loss = running_loss / len(train_data)
    training_acc.append(train_acc)
    training_loss.append(train_loss)
    print('Train --> Loss: {:.6f}, Acc: {:.6f}'.format(train_loss, train_acc))
    # train_loss.append(running_loss / total)
    # train_acc.append(100 * correct / total)

    # test_x, test_y = iter(test_loader).next()
    VGG19.eval()
    with torch.no_grad(): # 不需要算梯度，也不會進行反向傳播

        num_correct = 0
        eval_loss = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = VGG19(images)
            loss = loss_function(outputs, labels)
            eval_loss += loss.item() * labels.size(0)
            prob = torch.softmax(outputs, 1)
            _, predicted = torch.max(outputs, 1)
            num_correct += (predicted == labels).sum().item()

        val_acc = num_correct / len(val_data)
        val_loss = eval_loss / len(val_data)
        valing_acc.append(val_acc)
        valing_loss.append(val_loss)
        print('Val -->  Loss: {:.6f}, Acc: {:.6f}'.format(val_loss, val_acc))

        if val_acc > best_acc:
            best_model = copy.deepcopy(VGG19)

# save best model
torch.save(best_model, './VGG19_cifar10.pth')
with open('./hyperParameter.pkl', 'wb') as file:
        pkl.dump(hyperParameter, file)
file.close()
# write_pkl(hyperParameter, './data/hyperParameter.pkl')


# print(f'Epoch {epoch+1}, Loss: {running_loss / total:.3f}, Accuracy: {100 * eval_correct / total:.2f}%')
# val_loss.append(running_loss / total)
# val_acc.append(100 * eval_correct / total)

# torch.save(VGG19.state_dict(),'VGG19.pt')

# prediction = torch.argmax(VGG19(test_x), 1)
# acc = torch.eq(prediction, test_y)
# print('Accuracy: {:.2%}'.format((torch.sum(acc) / acc.shape[0]).item()))

title1 = 'Loss'
x = [i for i in range(1, EPOCH + 1)]
plt.plot(x, training_loss)
plt.plot(x, valing_loss)
plt.title(title1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.savefig('./vgg19_{}.jpg'.format(title1))
plt.show()

title2 = 'Accuracy'
x = [i for i in range(1, EPOCH + 1)]
plt.plot(x, training_acc)
plt.plot(x, valing_acc)
plt.title(title2)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train_acc', 'val_acc'], loc='upper left')
plt.savefig('./vgg19_{}.jpg'.format(title2))
plt.show()