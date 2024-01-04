import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 4),
        )

        self.learning_rate = 1e-3

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)

        self.loss = torch.nn.MSELoss(reduction = 'sum')

        self.loss_list = []

    def forward(self, x):
        x = torch.tensor(x, dtype = torch.float32)
        x = torch.flatten(x)
        logits = self.model(x)

        return logits

    def run_train(self, X, Y, epoch):

        self.model.train()

        for i in range(epoch):
            for batch, (x, y) in enumerate(zip(X, Y)):
                pred = self.forward(x)
                loss = self.loss(pred, torch.tensor(y, dtype = torch.float32))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if batch % 100 == 0:
                    loss = loss.item()
                    self.loss_list.append(loss)
                    print(loss)
        
    def plot_loss(self):
        x = range(len(self.loss_list))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(x, self.loss_list)
        plt.show()
    
if __name__ == '__main__':
    X = np.random.rand(100, 2)
    Y = np.concatenate((np.sin(X), np.cos(X)), axis = 1)
    print(X.shape, Y.shape)
    print(X, Y)
    
    epoch = 100

    net = Net()
    net.run_train(X, Y, epoch)
    net.plot_loss()