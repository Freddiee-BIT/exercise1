# -*- coding:utf-8 -*-
import torch
import pickle
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import gradio as gr
import matplotlib.pyplot as plt


class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.data_file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        else:
            self.data_file_names = ['test_batch']

        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for file_name in self.data_file_names:
            file_path = os.path.join(self.root, file_name)
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            if self.train:
                self.images.append(data[b'data'])
                self.labels.extend(data[b'labels'])
            else:
                self.images = data[b'data']
                self.labels = data[b'labels']

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
root_dir = 'cifar-10-python.tar/cifar-10-python/cifar-10-batches-py'

train_dataset = CIFAR10Dataset(root=root_dir, train=True, transform=train_transform)
test_dataset = CIFAR10Dataset(root=root_dir, train=False, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
if torch.cuda.is_available():
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

print('---------- Train Start ----------')
epochs = 50
epoch_loss = []
epoch_acc = []
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    cur_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(inputs)

        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        train_correct = (predicted == labels).sum()
        train_acc += train_correct.item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        cur_loss += loss.item()
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            running_loss = 0.0
    epoch_loss.append(cur_loss)
    print("Train Loss: {:.6f}, Acc: {:.6f}".format(cur_loss / 50000, train_acc / 50000))

    net.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for image, label in test_loader:
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()

        output = net(image)
        loss = criterion(output, label)
        eval_loss += loss.item() * label.size(0)
        _, predicted = torch.max(output, 1)
        eval_correct = (predicted == label).sum()
        eval_acc += eval_correct.item()
    print("Test Loss: {:.6f}, Acc: {:.6f}".format(eval_loss / 10000, eval_acc / 10000))
    epoch_acc.append(eval_acc / 10000)

print('----------Finished Training----------')

torch.save(net.state_dict(), 'model1.pth')
plt.figure(figsize=(10, 5))
epochs_list = list(range(1, epochs + 1))

plt.subplot(2, 1, 1)
plt.plot(epochs_list, epoch_loss, 'bo-', label='Loss')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(epochs_list, epoch_acc, 'r^-', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_plot1.jpg')


def preprocess(image):
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)
    return image

def visualize_loss_and_classify_image(image):
    input = preprocess(image).to('cuda:0')
    output = net(input)
    _, predicted = torch.max(output, 1)
    labels = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    prediction = labels[predicted]

    return prediction, 'training_plot1.jpg'

iface = gr.Interface(fn=visualize_loss_and_classify_image,  inputs=gr.Image(type='pil'), outputs=["text", "image"], title="Loss Visualization & Image Classification")

iface.launch()

