import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import time
import numpy
from PIL import Image
#import cv2
timelist = []
layer_num = 20
for i in range(layer_num):
    timelist.append([])

# 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            # 64
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            # 64
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            # M
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            # 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            # 128
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            # M
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv7 = nn.Sequential(
            # 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            # 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            # 256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            # M
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv11 = nn.Sequential(
            # 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            # 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            # 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Sequential(
            # M
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv15 = nn.Sequential(
            # 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv16 = nn.Sequential(
            # 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv17 = nn.Sequential(
            # 512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv18 = nn.Sequential(
            # M
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv19 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1)
        )

        self.convs = nn.Sequential(self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7,
                                   self.conv8, self.conv9, self.conv10, self.conv11, self.conv12, self.conv13,
                                   self.conv14, self.conv15, self.conv16, self.conv17, self.conv18, self.conv19)

        self.classifier = nn.Linear(450560, 10)

    def forward(self, tensor_x):

        print(tensor_x.shape)

        time0 = time.time()
        y = self.conv1(tensor_x)
        time2 = time.time()
        timelist[0].append(time2-time0)
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

        time1 = time.time()
        y = self.conv6(y)
        time2 = time.time()
        timelist[5].append(time2 - time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv7(y)
        time2 = time.time()
        timelist[6].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv8(y)
        time2 = time.time()
        timelist[7].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv9(y)
        time2 = time.time()
        timelist[8].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv10(y)
        time2 = time.time()
        timelist[9].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv11(y)
        time2 = time.time()
        timelist[10].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv12(y)
        time2 = time.time()
        timelist[11].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv13(y)
        time2 = time.time()
        timelist[12].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv14(y)
        time2 = time.time()
        timelist[13].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv15(y)
        time2 = time.time()
        timelist[14].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv16(y)
        time2 = time.time()
        timelist[15].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv17(y)
        time2 = time.time()
        timelist[16].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv18(y)
        time2 = time.time()
        timelist[17].append(time2-time1)
        print(y.shape)

        time1 = time.time()
        y = self.conv19(y)
        time2 = time.time()
        timelist[18].append(time2-time1)
        print(y.shape)

        y = y.view(y.size(0), -1)
        print(y.shape)

        time1 = time.time()
        y = self.classifier(y)
        time2 = time.time()
        timelist[19].append(time2 - time1)
        print(time)

        return y

    def head(self, tensor_x, clip_point=3):
        y = self.convs[:clip_point](tensor_x)

        return y

    def tail(self, tensor_x, clip_point=3):
        bs = tensor_x.shape[0]
        y = self.convs[clip_point:](tensor_x)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)
        return y

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
    ])

inputs = Image.open(r"../../data/test.jpg")
inputs = transform(inputs).unsqueeze(0)  # [1, 3, 720, 1280]

net = VGG16()
for i in range(10):
    outputs = net(inputs)

for i in range(layer_num):
    print("{}:{:.6f}".format(i, numpy.mean(timelist[i])))