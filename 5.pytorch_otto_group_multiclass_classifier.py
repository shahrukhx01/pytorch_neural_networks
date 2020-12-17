import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, nn
import numpy as np
import torch
import matplotlib.pyplot as plt


class OttoGroupDataset(Dataset):
    ## load dataset into memory using pandas then pass it to pytorch via numpy
    def __init__(self, data_path):

        ## read file into memory
        df = pd.read_csv(data_path)
  
        if('target' in df.columns):
            ## convert string labels to float labels for training covenience
            target_map = {"Class_{}".format(i+1):float(i) for i in range(9)}
            df.target = df.target.map(target_map)

            ## set target variable into y tensor
            self.y_data = from_numpy(df.target.values.astype(np.int_) )

        ##df.drop(['target', 'id'], inplace=True, axis=1)

        ## set predictor variables into x tensor
        self.x_data = from_numpy(df[[col for col in df.columns if col not in ['target', 'id']]].values.astype(np.float32))

        self.len = df.shape[0]
         

    ## return the number of examples in dataset
    def __len__(self):
        return self.len

    ## get specific item by index
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index]


## load train data into memory
train_dataset = OttoGroupDataset('datasets/otto_group_data/train.csv')

## wrap dataloader over dataset
train_loader = DataLoader(dataset= train_dataset, 
            batch_size=512,
            shuffle=True)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        ## define the architecture of the network
        self.network = nn.Sequential(
          nn.Linear(93,256),
          nn.ReLU(),
          nn.Linear(256,128),
          nn.ReLU(),
          nn.Linear(128,64),
          nn.ReLU(),
          nn.Linear(64,32),
          nn.ReLU(),
          nn.Linear(32,16),
          nn.ReLU(),
          nn.Linear(16,9),
          nn.Softmax()
        )
    
    ## complete the forward pass
    def forward(self, x):
        y_pred = self.network(x)
        return y_pred

## create model instance
model = Model()

## Since it has builtin Softmax + CrossEntropy which would help in multi-class classification
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


losses = []
## training loop 
for epoch in range(10):
    epoch_loss = []
    for i,data in enumerate(train_loader):
        x_data, y_data = data

        ## perform the forward pass
        y_pred = model(x_data)

        ## compute loss
        loss = criterion(y_pred, y_data)

        print("Epoch/iteration: {}/{} with loss {}".format(epoch, i, loss.item()))
        epoch_loss.append(loss.item())
    
    losses.append(sum([loss/ len(epoch_loss) for loss in epoch_loss]))


#plt.plot(losses)
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.show()
    