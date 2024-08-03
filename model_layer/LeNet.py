import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import numpy
from PIL import Image
#import cv2
timelist = []
for i in range(6):
    timelist.append([])

class LeNet(nn.Module):
    def __init__(self, num_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            # 3, 224, 224 -> 64, 27, 27
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            # 64, 27, 27 -> 192, 13, 13
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            # 192, 13, 13 -> 384, 13, 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.conv4 = nn.Sequential(
            # 384, 13, 13 -> 256, 13, 13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            # 256, 13, 13 -> 256, 6, 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 45 * 80, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )

        self.convs = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.fc)

    def forward(self, tensor_x):
        bs = tensor_x.shape[0]
        print(tensor_x.shape)
        time1 = time.time()
        y = self.conv1(tensor_x)
        time2 = time.time()
        timelist[0].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv2(y)
        time2 = time.time()
        timelist[1].append(time2 - time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv3(y)
        time2 = time.time()
        timelist[2].append(time2 - time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv4(y)
        time2 = time.time()
        timelist[3].append(time2 - time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv5(y)
        time2 = time.time()
        timelist[4].append(time2 - time1)
        print(y.shape)

        y = y.view(bs, -1)
        print(y.shape)

        time1 = time.time()
        y = self.fc(y)
        time2 = time.time()
        timelist[5].append(time2 - time1)
        print(y.shape)

        return y

    def head(self, tensor_x, clip_point=3):
        y = self.convs[:clip_point](tensor_x)

        return y

    def tail(self, tensor_x, clip_point=3):
        bs = tensor_x.shape[0]
        y = self.convs[clip_point:](tensor_x)
        y = y.view(bs, -1)
        y = self.fc(y)
        return y



transform = transforms.Compose([
        transforms.ToTensor(),
    ])
'''
dataset = torchvision.datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
inputs, target = 0, 0
for j, data in enumerate(loader):
    inputs, target = data
'''

inputs = Image.open(r"../../data/test.jpg")

inputs = transform(inputs).unsqueeze(0)  # [1, 3, 720, 1280]

net = LeNet()

for i in range(10):
    outputs = net(inputs)

for i in range(6):
    # print(timelist[i])
    print(numpy.mean(timelist[i]))