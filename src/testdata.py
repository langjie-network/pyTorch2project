import PIL.Image
import torch
import PIL.Image
import numpy as np
import os
import config
rootPath=config.rootPath



def load_data(fileName):
    xs = []
    ys = []
    #读取图片信息
    x = PIL.Image.open(rootPath+fileName)
    x=x.resize((32,32))
    #转矩阵,数值压缩到0-1之间
    x = torch.FloatTensor(np.array(x)) / 255
    #变形,把通道放前面
    #[32, 32, 3] -> [3, 32, 32]
    x = x.permute(2, 0, 1)
    #y来自文件名的第一个字符
    xs.append(x)
    ys.append(-1)
    return xs, ys

xs,ys=load_data('data2/testdata/testB1.jpg')

class MyTestDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(xs)
    def __getitem__(self, i):
        return xs[i], ys[i]
