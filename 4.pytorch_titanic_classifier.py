import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, nn
import numpy as np
import torch
import matplotlib.pyplot as plt

class TitanicDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('titanic.csv')
        df = df[['Age','Fare','Survived']]

        df = df.fillna(-1) ## fill missing values

        self.len = df.shape[0]
        
        self.y_data = from_numpy(np.asmatrix(df['Survived'].values).T.astype(np.float32) )
        df.drop(['Survived'], inplace=True, axis=1)

        self.x_data = from_numpy(df.values.astype(np.float32) )
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index]



dataset = TitanicDataset()

dataloader = DataLoader(dataset= dataset, 
            batch_size=32,
            shuffle=True)


## create a single hidden layer neural network
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.network = nn.Sequential(
          nn.Linear(2,4),
          nn.ReLU(),
          nn.Linear(4,2),
          nn.ReLU(),
          nn.Linear(2,1),
          nn.Sigmoid()
        )
    
    def forward(self, x):
        y_pred = self.network(x)
        return y_pred


model = Model()
criterion = nn.BCELoss(reduction='mean') ## Since we are doing binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


losses = []
## training loop
for epoch in range(100):
    epoch_loss = []
    for i,data in enumerate(dataloader):
        
        x_data, y_data = data
        
        ## complete the forward pass on the network
        y_pred = model(x_data)
        
        ## compute loss on the batch after forward pass
        loss = criterion(y_pred, y_data)
        
        ## reset loss gradients and compute backward pass and adjust weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("Epoch/iteration: {}/{} with loss {}".format(epoch, i, loss.item()))
        epoch_loss.append(loss.item())

    losses.append(sum([loss/ len(epoch_loss) for loss in epoch_loss]))


plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    


print(model(torch.tensor([[25.0, 30.0]])))