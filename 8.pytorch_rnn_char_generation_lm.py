import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np


class LanguageModelRNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size, sequence_length, num_directions, batch_size, num_classes ):
        super(LanguageModelRNN, self).__init__()

        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.num_directions = num_directions
        self.num_classes = num_classes
        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)    

    def create_data(self):
        self.idx2char = ['h', 'i', 'e', 'l', 'o']

        # Teach hihell -> ihello
        x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
        x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
                    [0, 1, 0, 0, 0],   # i 1
                    [1, 0, 0, 0, 0],   # h 0
                    [0, 0, 1, 0, 0],   # e 2
                    [0, 0, 0, 1, 0],   # l 3
                    [0, 0, 0, 1, 0]]]  # l 3

        y_data = [1, 0, 2, 3, 3, 4]    # ihello

        # As we have one batch of samples, we will change them to variables only once
        self.inputs = Variable(torch.Tensor(x_one_hot))
        self.labels = Variable(torch.LongTensor(y_data))


    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size) for batch_first=True
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Reshape input
        x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)

        out, _ = self.rnn(x, h_0)
        return out.view(-1, self.num_classes)
    
    def train(self, model):
        self.create_data()
        
        
        # Set loss and optimizer function
        # CrossEntropyLoss = LogSoftmax + NLLLoss
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.1)

        # Train the model
        for epoch in range(100):            
            outputs = model(self.inputs)
            optimizer.zero_grad()
            print(outputs.shape)
            loss = criterion(outputs, self.labels)
            loss.backward()
            optimizer.step()
            _, idx = outputs.max(1)
            idx = idx.data.numpy()
            result_str = [self.idx2char[c] for c in idx.squeeze()]
            print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.item()))
            print("Predicted string: ", ''.join(result_str))



if __name__ == '__main__':
    
    lm_rnn = LanguageModelRNN(num_layers=1, hidden_size=5, input_size=5, 
                    sequence_length=6, num_directions=1, batch_size=1, 
                    num_classes=5 )
    
    lm_rnn.train(lm_rnn)
    
