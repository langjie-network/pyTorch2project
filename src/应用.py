
import torch
import  config
from Model import Model
rootPath=config.rootPath
import testdata

testdataset = testdata.MyTestDataset()
#数据加载器
loader = torch.utils.data.DataLoader(dataset=testdataset,
                                     batch_size=1,
                                     shuffle=True,
                                     drop_last=True);

x,y = next(iter(loader))


def test():
    model = torch.load(rootPath+'/model/15.model')
    model.eval()
    #取概率最大的那个
    out =model(x).argmax(dim=1)
    print(out)


test()