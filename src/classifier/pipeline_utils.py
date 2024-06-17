import time
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json 
from collections import defaultdict


class Trainer:
    def __init__(self, 
                 model: str, 
                 dataloaders, 
                 criterion, 
                 optimizer,
                 device: str = "cuda:0", 
                 num_epochs: int = 25,
                 train_only: bool = False,
                 is_inception: bool = False,
                 verbose: bool = True
                ):
        
        self.model = model
        self.device = device 
        self.dataloaders = dataloaders 
        self.criterion = criterion 
        self.optimizer = optimizer 
        self.num_epochs =  num_epochs
        self.is_inception = is_inception
        self.verbose =verbose

        # stats
        self.training_history = {'train': defaultdict(list), 'validation': defaultdict(list)}
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0

        self.model = self.model.to(device)
        self.steps = ['train', 'validation'] if not train_only else ['train']
    
    def train(self):
        since = time.time() 
        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f'Epoch {epoch}/{self.num_epochs - 1}\n{"-"*10}')
    
            # Each epoch has a training and validation step
            for step in self.steps:
                if step == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0
    
                # Iterate over data.
                for inputs, labels in tqdm(self.dataloaders[step]):  # data in batches
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(step == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if self.is_inception and step == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = self.model(inputs)
                            loss1 = self.criterion(outputs, labels)  # loss is calculated before softmax
                            loss2 = self.criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, labels) # PyTorch loss fucntion takes only integer labels, no need one-hot encoding as it is included!
    
                        # backward + optimize only if in training step
                        if step == 'train':
                            loss.backward()
                            self.optimizer.step()
    
                    # statistics
                    #_, preds = torch.max(outputs, 1)
                    preds = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
    
                epoch_loss = running_loss / len(self.dataloaders[step].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[step].dataset)
                self.training_history[step]['loss'].append(epoch_loss)
                self.training_history[step]['acc'].append(float(epoch_acc.cpu().numpy()))
                if self.verbose:
                    print('{} loss: {:.4f} acc: {:.4f}'.format(step, epoch_loss, epoch_acc))
                
                
                # update the model with the best performance (deep copy the model)
                if step == 'validation' and epoch_acc > self.best_acc:
                    self.best_acc = epoch_acc
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())

        time_elapsed = time.time() - since
        if self.verbose:
            print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(self.best_acc))
    
        # return the best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.training_history

    def plot_history(self, xticks: int = 5, save_fig: bool=True, out_name: str=None):
        x = [i+1 for i in range(len(self.training_history['train']['loss']))]
        colors = {'loss': 'tab:red', 'acc': 'tab:blue'}
        linestyles = {'train': 'solid', 'validation': 'dashed'}
        fig, ax1 = plt.subplots()
        ax1.xaxis.set_major_locator(plt.MaxNLocator(xticks))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('loss', color=colors['loss'])
        ax1.tick_params(axis='y', labelcolor=colors['loss'])
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('accuracy', color=colors['acc'])  # we already handled the x-label with ax1
        ax2.tick_params(axis='y', labelcolor=colors['acc'])
        for step in self.steps:
            if step in self.training_history:                
                ax1.plot(x, self.training_history[step]['loss'], color=colors['loss'], ls=linestyles[step], label='loss')
                ax2.plot(x, self.training_history[step]['acc'], color=colors['acc'], ls=linestyles[step], label='acc')
                
        ax1.legend(['train', 'validation'], loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        fig.tight_layout()  # otherwise the right y-label is slightly clipped        
        plt.show()

        if save_fig:
            fig.savefig(f'{out_name}')


    def save_results(self, output_path: str = None, model_name: str = None, weights_only: bool = False, save_history: bool = True):
        out_name = output_path + model_name
        if weights_only:
            torch.save(self.model.state_dict(), out_name+'.pth')
        else:
            torch.save(self.model, out_name+'.pth')

        if save_history:
            with open(f"{out_name+'history'}.json", "w") as file:
                json.dump(self.training_history, file)
                file.close()


class Evaluation:
    def __init__(self, 
                 model, 
                 dataloader, 
                 threshold: float = 0.5, 
                 classification: str = 'real', 
                 device: str = "cuda:0"):
        
        self.model = model.to(device)
        self.dataloader = dataloader
        self.threshold = threshold
        self.device = device
        
        classes = {'fake': 0, 'real': 1}
        self.categ = classes[classification]

    def evaluate(self):
        self.model.eval()
        acc = FP = TP = 0
        i = 0
        for images,labels in tqdm(self.dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            probs = nn.Softmax(dim=self.categ)(outputs)
            masks = ((probs > self.threshold)==True).nonzero()
            preds = masks[:,1]
            inds = masks[:,0]
            truths = labels[inds]
            acc += torch.sum(preds==truths)
            

            predc1 = (preds==1).nonzero()
            TP += torch.sum(truths[predc1]==1)
            FP += torch.sum(truths[predc1]!=1)

        print(f'avg acc: {acc/len(self.dataloader.dataset)}, prec: {TP/(TP+FP)}')



class Inference:
    def __init__(self, 
                 model, 
                 dataloader, 
                 threshold: float = 0.5, 
                 classification: str = 'real', 
                 device: str = 'cpu',
                 filepaths: list=[]):
        
        self.model = model
        self.test_data = iter(dataloader)
        self.threshold = threshold
        self.device = device
        self.filepaths = filepaths
        
        classes = {'fake': 0, 'real': 1}
        self.categ = classes[classification]

    def next_batch(self):
        self.model.eval()
        batch_data = next(self.test_data)
        inputs, ids = batch_data
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        probs = nn.Softmax(dim=self.categ)(outputs)

        mask = probs[:,self.categ] > self.threshold
        if len(self.filepaths):
            f_paths = [self.filepaths[i] for i in ids[mask.cpu()].numpy()]
            n = len(mask.nonzero())
        
            if len(f_paths) == 1:
                fig, ax = plt.subplots(figsize=(24, 24))
                ax.imshow(np.transpose(inputs[i][0,:].cpu(), (1,2,0)))
                print(f_paths[0])
        
            elif len(f_paths) > 1:
                _, axs = plt.subplots(len(mask.nonzero()), 1, figsize=(24, 24))
                #axs = axs.flatten()
                for i, f, ax in zip(mask.nonzero(), f_paths, axs):
                    ax.imshow(np.transpose(inputs[i][0,:].cpu(), (1,2,0)))
                    print(f)
                plt.show()