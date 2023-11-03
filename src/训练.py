
import torch
import  config
from Model import Model
rootPath=config.rootPath
import traindata
traindataset = traindata.MyTrainDataset()


model = Model()

loader = torch.utils.data.DataLoader(dataset=traindataset,
                                     batch_size=8,
                                     shuffle=True,
                                     drop_last=True);



def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fun = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(1000):
        for i, (x, y) in enumerate(loader):
            out = model(x)
            loss = loss_fun(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 100 == 0 and epoch % 100==0:
                acc = (out.argmax(dim=1) == y).sum().item() / len(y)
                print(epoch, i, loss.item(), acc)

    torch.save(model, rootPath+'model/15.model')
    print("over............")


train()