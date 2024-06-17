import glob
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import models
from transformers import ViTModel, ViTConfig
import torchvision.datasets as dset
import torchvision.utils as vutils

from torch.nn import Module, Sequential
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, Sigmoid, Flatten, Unflatten, Linear

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Callable, Optional

#------------------ models ----------------------
def set_parameter_requires_grad(model, feature_extracting):
    """when feature_extracting=True, only update the output head
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# A pretrained ViT model from transformers
class ViT(nn.Module):
    """
    By default, it contains the embedding layer, 12 ViTEncoder units, and the output layer (head). 
    Each ViTEncoder unit has an attention layer, the feedforward layer (itermediate), and LayerNorm.
    """

    def __init__(self, 
                 config=ViTConfig(), 
                 num_labels=20, 
                 feature_extract=False,
                 model_checkpoint='google/vit-base-patch16-224-in21k'):

        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        set_parameter_requires_grad(self.vit, feature_extract)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )

    def forward(self, x):

        x = self.vit(x)['last_hidden_state']
        # Use the embedding of [CLS] token
        output = self.classifier(x[:, 0, :])

        return output


class myModels:
    def __init__(self, model_name, num_classes, feature_extract, use_pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
    
    def model(self):
        return initialize_model(self.model_name, self.num_classes, self.feature_extract, self.use_pretrained)
    
    @staticmethod
    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0
        if model_name == "vit_p16":
            """Vision Transformer, 16x16 patches
            """
            model_ft = ViT(num_labels=num_classes, feature_extract=feature_extract)
            input_size = 224
            
        elif model_name == 'vit_p32':
            """Vision Transformer, 32x32 patches
            """
            model_ft = ViT(num_labels=num_classes, feature_extract=feature_extract, model_checkpoint='google/vit-base-patch32-224-in21k')
            input_size = 224
            
        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes) # modified the fully connected layer and only update it
            input_size = 224
    
        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
    
        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224
    
        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224
    
        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
            input_size = 224
    
        elif model_name == "inception":
            """ Inception v3 
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299
    
        else:
            print("Invalid model name, exiting...")
            exit()
        
        return model_ft, input_size



#------------------ data ------------------------
# create customized dataset class
class myDataset(torch.utils.data.Dataset):
    '''
    Args:
        folder      str, folder path
        format      str, formats supported by PIL
        label       int, default 0
        transform   callable, if not given return PIL image

    Return:
        tensor or PIL image     tuple, (image, label)
    '''
    def __init__(self, 
                 root=None, 
                 format=None, 
                 label=None, 
                 transform=None,
                 return_filepath=False):
        
        _file_path = root+'/*.'+format if format else root+'/*'
        self.dataset = glob.glob(_file_path) #filenames matching full path
        self.label = label if label else 0
        self.transform = transform
        self.return_filepath = return_filepath

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        if self.return_filepath:
            self.label = idx
        
        if self.transform:
            return self.transform(Image.open(data).convert('RGB')), self.label
        else:
            return Image.open(data).convert('RGB'), self.label


class myDataloader:
    """Create datasets and return dataloaders loading these datasets.
    """
    def __init__(self, 
                 root: str = None, 
                 batch_size: int = 16, 
                 transform: Callable = None,
                 shuffle: bool = True,
                 split: list = None,
                 num_workers: int = 1,
                 from_subfolders: bool = False,
                 format = None, 
                 label = None,
                 return_filepath = False):
        
        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        self.shuffle = shuffle
        self.split = split
        self.workers = num_workers
        self.from_subfolders = from_subfolders

        self.format = format
        self.label = label
        self.return_filepath = return_filepath
    
    @staticmethod
    def create_datasets(root: str = None, 
                        transform: Callable = None,
                        format: str = None,
                        label: str = None,
                        split: list = None,
                        from_subfolders: bool = False,
                        return_filepath: bool = False):
        
        if from_subfolders:
            dataset = dset.ImageFolder(root=root, transform=transform)
        else:
            dataset = myDataset(root=root, 
                                transform=transform, 
                                format=format, 
                                label=label, 
                                return_filepath=return_filepath)

        return dataset if not split else torch.utils.data.random_split(dataset, split)


    def dataloader(self):
        ds = self.create_datasets(root = self.root, 
                                  transform = self.transform,
                                  format = self.format,
                                  label= self.label,
                                  split = self.split,
                                  from_subfolders = self.from_subfolders,
                                  return_filepath = self.return_filepath)
        
        return (torch.utils.data.DataLoader(d, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.workers) for d in ds)

    @staticmethod
    def plot_dataloader(dataloader, device):
        """Plot the dataloader whole batch 
        """
        real_batch, y = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



class cnnAutoencoder(Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.encoder = Sequential(
            Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(64* (input_shape[1] // 4) * (input_shape[2] // 4), latent_dim)
        )
        self.decoder = Sequential(
            Linear(latent_dim, 64 * (input_shape[1] // 4) * (input_shape[2] // 4)),
            Unflatten(1, (64, input_shape[1] // 4, input_shape[2] // 4)),
            ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, 
                            output_padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=32, out_channels=input_shape[0], kernel_size=3, stride=2, 
                            padding=1, output_padding=1),
            #Sigmoid()
        )
    
    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return z



