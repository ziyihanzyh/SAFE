import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy
from PIL import Image

timelist = []
layer_num = 9
for i in range(layer_num):
    timelist.append([])

'''
modified to fit dataset size
'''
NUM_CLASSES = 10


class AlexNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.convs = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8)

    def forward(self, x):
        time1 = time.time()
        y = self.conv1(x)
        time2 = time.time()
        timelist[0].append(time2 - time1)

        time1 = time.time()
        y = self.conv2(y)
        time2 = time.time()
        timelist[1].append(time2 - time1)

        time1 = time.time()
        y = self.conv3(y)
        time2 = time.time()
        timelist[2].append(time2 - time1)

        time1 = time.time()
        y = self.conv4(y)
        time2 = time.time()
        timelist[3].append(time2 - time1)

        time1 = time.time()
        y = self.conv5(y)
        time2 = time.time()
        timelist[4].append(time2 - time1)

        time1 = time.time()
        y = self.conv6(y)
        time2 = time.time()
        timelist[5].append(time2 - time1)

        time1 = time.time()
        y = self.conv7(y)
        time2 = time.time()
        timelist[6].append(time2 - time1)

        time1 = time.time()
        y = self.conv8(y)
        time2 = time.time()
        timelist[7].append(time2 - time1)

        y = y.view(y.size(0), 256 * 2 * 2)
        time1 = time.time()
        y = self.classifier(y)
        time2 = time.time()
        timelist[8].append(time2 - time1)

        return y

    def head(self, tensor_x, clip_point=3):
        y = self.convs[:clip_point](tensor_x)

        return y

    def tail(self, tensor_x, clip_point=3):
        bs = tensor_x.shape[0]
        y = self.convs[clip_point:](tensor_x)
        y = y.view(y.size(0), 256 * 2 * 2)
        y = self.classifier(y)
        return y

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
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


net = AlexNet()


for i in range(10):
    outputs = net(inputs)

for i in range(layer_num):
    # print(timelist[i])
    print(numpy.mean(timelist[i]))