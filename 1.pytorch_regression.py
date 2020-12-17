import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1,1)

    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

x_data = torch.tensor([[12.0],[13.0],[15.0],[16.0]])
y_data = torch.tensor([[24.0],[26.0],[30.0],[32.0]])


criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)


for epoch in range(500):
    y_pred = model(x_data)

    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch {} | The loss: {}'.format(epoch, loss.item()))




print(
    model(torch.tensor([[20.0]]).data[0])
    )


    





























