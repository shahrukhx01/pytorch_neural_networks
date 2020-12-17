import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor, nn
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


## most of this code comes from pytorch CIFAR10 tutorial
## https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

PATH = './cifar_net.pth'

class CIFAR10Dataset():

    ## Pytorch has built-in support for CIFAR10 dataset, so we don't need to write custom dataloader here
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) ## to normalize pixel values

        trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def view_examples(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        # show images
        print(images.shape)
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))




## Now let's create our model

class CIFAR10Model(nn.Module):

    def __init__(self):

        super(CIFAR10Model, self).__init__()

        ## here we define components of our network
        ## I'd copy the architecture given on pytorch's site
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        ## write the forward pass of the network
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        ## here you have to transform image from 3d to 2 dimensions to feed into fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(dataset, model):
    ## define loss function, since we have multi-class classification problem so CrossEntropy is a good option
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ## finally we write our train loop
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataset.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    
    torch.save(model.state_dict(), PATH)
    print('Model Saved.')


def eval(dataset, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset.testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

## freeze multiprocessing 
## this is guard for MACOS you can omit this if you're not running on MACOS
def run():
    torch.multiprocessing.freeze_support()

    ## download dataset
    cifar10_dataset = CIFAR10Dataset()
    ## view few images from trainset
    ##cifar10_dataset.view_examples()

    ## run the train loop
    cifar_model = CIFAR10Model()
    ##train(cifar10_dataset, cifar_model)

    ##lets load the model and evaluate performance of our model
    cifar_model.load_state_dict(torch.load(PATH))
    eval(cifar10_dataset, cifar_model)


if __name__ == '__main__':
    run()

