import torch
import torch.nn.functional as F

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(1,1)


    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.], [0.], [1.], [1.]])

for epoch in range(500):
    y_pred = model(x_data)

    print(y_pred)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    print("Loss is ", loss.item())


hour_var = torch.tensor([[60.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())

