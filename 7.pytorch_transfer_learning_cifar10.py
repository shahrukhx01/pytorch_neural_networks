from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



## Transfer learning script from pytorch official tutorial
## https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
## dataset: https://download.pytorch.org/tutorial/hymenoptera_data.zip

class TransferLearningCNN():
    def __init__(self, data_dir, num_classes, batch_size, 
                        num_epochs, feature_extract, criterion,
                        model_name, use_pretrained, optimizer_name, lr, momentum, dataset_name):
        
        ## initialize attributes for training and eval
        self.data_dir = data_dir 
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.feature_extract = feature_extract
        self.criterion = criterion
        self.model_name = model_name
        if (model_name == "inception"):
            self.is_inception = True
        else:
            self.is_inception = False
        self.model, self.input_size = self.initialize_model(use_pretrained=use_pretrained) 
        self.lr = lr
        self.momentum = momentum
        self.optimizer_name = optimizer_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Detect if we have a GPU available
        self.model = self.model.to(self.device) ## Send the model to GPU if available
        self.optimizer = self.configure_optimizer() ## now we create optimizer after creating and initializing our model
        self.dataset_name = dataset_name
        self.dataloaders = self.prepare_data() ## dictionary of dataloaders {'train':..., 'val':...}
    """
    Gather the parameters to be optimized/updated in this run. If we are
    finetuning we will be updating all parameters. However, if we are
    doing feature extract method, we will only update the parameters
    that we have just initialized, i.e. the parameters with requires_grad
    is True.
    """
    def configure_optimizer(self):

        params_to_update = self.model.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)

        # Observe that all parameters are being optimized
        optimizer_ft = None
        if self.optimizer_name == "SGD":
            optimizer_ft = optim.SGD(params_to_update, self.lr, self.momentum)
        
        return optimizer_ft

    def train_model(self):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if self.is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return val_acc_history


    """
    This helper function sets the .requires_grad attribute of the parameters in the model to 
    False when we are feature extracting. By default, when we load a pretrained model all of 
    the parameters have .requires_grad=True, which is fine if we are training from scratch or 
    finetuning. However, if we are feature extracting and only want to compute gradients for 
    the newly initialized layer then we want all of the other parameters to not require gradients. 
    This will make more sense later.
    """
    def set_parameter_requires_grad(self, model):
        if self.feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        
        return model


    ## initialize and reshape networks
    """When feature extracting, we only want to update the parameters of the last layer, 
    or in other words, we only want to update the parameters for the layer(s) we are reshaping. 
    Therefore, we do not need to compute the gradients of the parameters that we are not changing, 
    so for efficiency we set the .requires_grad attribute to False. This is important because by 
    default, this attribute is set to True. Then, when we initialize the new layer and by default 
    the new parameters have .requires_grad=True so only the new layer’s parameters will be updated. 
    When we are finetuning we can leave all of the .required_grad’s set to the default of True."""
    def initialize_model(self, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if self.model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = 224

        elif self.model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,self.num_classes)
            input_size = 224

        elif self.model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            model_ft.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = self.num_classes
            input_size = 224

        elif self.model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, self.num_classes)
            input_size = 224

        elif self.model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            model_ft = self.set_parameter_requires_grad(model_ft)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, self.num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,self.num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size
    
    def prepare_data(self):
        # Data augmentation and normalization for training
        # Just normalization for validation

        print("Initializing Datasets and Dataloaders...")

        dataloaders_dict = {}
        
        if self.dataset_name == "hymenoptera_data":
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
            # Create training and validation dataloaders
            dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
        
        
        elif self.dataset_name == "cifar10":


             transform = transforms.Compose([transforms.ToTensor(), 
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) ## to normalize pixel values
             trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10', train=True,
                                                    download=True, transform=transform)
             train_set, val_set = torch.utils.data.random_split(trainset, [40000, 10000])

             train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=2)
             val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2) 
             self.classes = ('plane', 'car', 'bird', 'cat',
                    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            

             dataloaders_dict = {'train': train_loader, 'val': train_loader}   

        return dataloaders_dict


    def imshow(self, img):
        ##img = img 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    def view_examples(self):
        # get some random training images
        dataiter = iter(self.dataloaders['train'])
        images, labels = dataiter.next()

        # show images
        print(images.shape)
        self.imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

## freeze multiprocessing 
## this is guard for MACOS you can omit this if you're not running on MACOS
def run():
    torch.multiprocessing.freeze_support()

    # Top level data directory. Here we assume the format of the directory conforms
    ##   to the ImageFolder structure
    data_dir = "./datasets/hymenoptera_data"

    ## Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "squeezenet"

    ## Number of classes in the dataset
    num_classes = 10

    ## Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # Number of epochs to train for
    num_epochs = 15

    ## Flag for feature extracting. When False, we finetune the whole model,
    ##   when True we only update the reshaped layer params
    feature_extract = True

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    
    ## create model object
    ft_model = TransferLearningCNN(data_dir=data_dir, num_classes=num_classes, batch_size=batch_size, 
                        num_epochs=num_epochs, feature_extract=feature_extract, criterion=criterion,
                        model_name=model_name, use_pretrained=True, optimizer_name="SGD", lr=0.001, momentum= 0.9,
                         dataset_name="cifar10")
    


    ## view train images
    #ft_model.view_examples()

    ## train model
    ft_model.train_model()
    


if __name__ == '__main__':
    
    run()
