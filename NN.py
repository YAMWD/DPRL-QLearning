import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  
from torch.nn.utils import clip_grad_norm_

class MyDataset(Dataset):  
    def __init__(self, data):  
        self.data = data  
  
    def __len__(self):  
        return len(self.data)  
  
    def __getitem__(self, idx):  
        return self.data[idx]  
    
    def addSample(self, x, y):
        self.data.append([x, y])

class Net(torch.nn.Module):
    def __init__(self, learning_rate, batch_size):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

        self.batch_size = batch_size

        self.learning_rate = learning_rate

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)

        self.loss = torch.nn.MSELoss(reduction = 'sum')
    
        data = [[np.array([0.0, 0.0]), 0.0]]

        # data = [[torch.tensor([0.0, 0.0]), torch.tensor(0.0)]]

        self.dataset = MyDataset(data)  

        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

        self.loss_list = []

    def addSample(self, x, y):
        self.dataset.addSample(x, y)

        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True)

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)

        logits = self.model(x)

        return logits

    def run_train(self, epoch):

        losses = []
        self.model.train()

        for i in range(epoch):
            for batch, (x, y) in enumerate(self.dataloader):
                pred = self.forward(x)
                loss = self.loss(pred, torch.tensor(y, dtype = torch.float32))

                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm = 1.0)  
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.item())
            
            self.loss_list.append(np.mean(losses))
        
    def plot_loss(self, name):
        x = range(len(self.loss_list))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        # plt.yscale('log')
        plt.plot(x, self.loss_list)
        plt.savefig(name, bbox_inches = 'tight')
    
if __name__ == '__main__':
    X = np.random.rand(100, 2)
    Y = np.mean(np.sin(X), axis = 1)
    print(X.shape, Y.shape)
    print(X, Y)

    epoch = 10000
    net = Net(1e-3, 32)
    for (x, y) in zip(X, Y):
        net.addSample(x, y)
    
    net.run_train(epoch)
    net.plot_loss('figs/test.pdf')