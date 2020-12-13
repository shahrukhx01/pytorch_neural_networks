import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class TitanicDataset(Dataset):
    def __init__(self):
        df = pd.read_csv('titanic.csv')
        df = df[['PassengerId','Age','Fare','Survived']]
        df = df.fillna(-1)
        self.len = df.shape[0]
        self.y_data = from_numpy(df['Survived'].values )
        df.drop(['Survived'], inplace=True, axis=1)
        self.x_data = from_numpy(df.values.astype(np.float32) )

        #['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
        #'Ticket', 'Fare', 'Cabin', 'Embarked']
        
        print(type(self.x_data), type(self.y_data))
        print(self.x_data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x_data[index, :], self.y_data[index]



dataset = TitanicDataset()

dataloader = DataLoader(dataset= dataset, 
            batch_size=32,
            shuffle=True)




for i,data in enumerate(dataloader):
    x,y = data
    print(x.shape, y.shape)