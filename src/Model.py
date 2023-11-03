import PIL.Image
import torch
import PIL.Image
import numpy as np
import os
import  config
rootPath=config.rootPath

#cnn神经网络
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #520的卷积层
        self.cnn1 = torch.nn.Conv2d(in_channels=3,
                                    out_channels=16,
                                    kernel_size=5,
                                    stride=2,
                                    padding=0)

        #311的卷积层
        self.cnn2 = torch.nn.Conv2d(in_channels=16,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        #710的卷积层
        self.cnn3 = torch.nn.Conv2d(in_channels=32,
                                    out_channels=128,
                                    kernel_size=7,
                                    stride=1,
                                    padding=0)

        #池化层
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        #激活函数
        self.relu = torch.nn.ReLU()

        #线性输出层
        self.fc = torch.nn.Linear(in_features=128, out_features=10)

    def forward(self, x):

        #第一次卷积,形状变化可以推演
        #[8, 3, 32, 32] -> [8, 16, 14, 14]
        x = self.cnn1(x)
        x = self.relu(x)

        #第二次卷积,因为是311的卷积,所以尺寸不变
        #[8, 16, 14, 14] -> [8, 32, 14, 14]
        x = self.cnn2(x)
        x = self.relu(x)

        #池化,尺寸减半
        #[8, 32, 14, 14] -> [8, 32, 7, 7]
        x = self.pool(x)

        #第三次卷积,因为核心是7,所以只有一步计算
        #[8, 32, 7, 7] -> [8, 128, 1, 1]
        x = self.cnn3(x)
        x = self.relu(x)

        #展平,便于线性计算,也相当于把图像变成向量
        #[8, 128, 1, 1] -> [8, 128]
        x = x.flatten(start_dim=1)

        #线性计算输出
        #[8, 128] -> [8, 10]
        return self.fc(x)

