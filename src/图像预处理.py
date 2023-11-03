import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import  config
from Model import Model
rootPath=config.rootPath


# 读取图片
image = Image.open(rootPath+'data2/testdata/A/A0.jpg')

# 定义预处理转换
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # 重设大小为32x32
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
])

# 进行预处理
processed_image = preprocess(image)

# 查看预处理后的图像大小
print(processed_image.size())

torchvision.utils.save_image(processed_image, rootPath+'data2/testdata/A/A8.jpg')

print("图像已保存到")
