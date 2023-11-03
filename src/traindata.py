import PIL.Image
import torch
import PIL.Image
import numpy as np
import os
import config
rootPath=config.rootPath



def load_data(dir,label):
    xs = []
    ys = []
    #遍历文件夹下的所有文件
    for filename in os.listdir(rootPath+dir):
        print(filename)
        #只要图片,过滤一些无关的文件
        if not filename.endswith('.jpg'):
            continue
        #读取图片信息
        x = PIL.Image.open(rootPath+dir+'/%s' % filename)
        x=x.resize((32,32))
        #转矩阵,数值压缩到0-1之间
        x = torch.FloatTensor(np.array(x)) / 255
        #变形,把通道放前面
        #[32, 32, 3] -> [3, 32, 32]
        x = x.permute(2, 0, 1)
        #y来自文件名的第一个字符
        y = label
        xs.append(x)
        ys.append(y)
    return xs, ys




xs_A,ys_A=load_data('data2/traindata/A',0)
xs_B,ys_B=load_data('data2/traindata/B',1)

xs=[]
ys=[]
xs.extend(xs_A)
xs.extend(xs_B)
ys.extend(ys_A)
ys.extend(ys_B)



class MyTrainDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(xs)
    def __getitem__(self, i):
        return xs[i], ys[i]
