import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda
import numpy as np
import os
import logging

class Trainer(object):
    """
    Trains a PyTorch neural network with a given input and multilabel dataset using multiple cells simultaneously
    """
    def __init__(self, net, inputs_volume, labels_volume, checkpoint_dir, checkpoint_period=5000, 
                 logger_dir=None, checkpoint=None, optimizer=None, criterion=None, 
                 max_epochs=10, gpu_device=None, validation_split=0.2):
        """
        Sets up the parameters for training

        :param net: A PyTorch neural network
        :param inputs_volume: A list containing training and validation inputs
        :param labels_volume: A list containing training and validation corresponding multilabels
        :param checkpoint_dir: The directory to save checkpoints
        :param checkpoint_period: The number of iterations between checkpoints
        """
        self.max_epochs = max_epochs

        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(0))) # fix it

        if optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.inputs_volume = inputs_volume
        self.labels_volume = labels_volume 
        
        if not os.path.isdir(checkpoint_dir):
            raise IOError("{} is not a valid directory".format(checkpoint_dir))
        
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period
        self.max_accuracy = 0
        
        if logger_dir is not None and not os.path.isdir(logger_dir):
            raise IOError("{} is not a valid directory".format(logger_dir))
            
        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)    
        
        if logger_dir:
            file_handler = logging.FileHandler(os.path.join(logger_dir,
                                                            "training.log"))
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler) 
                
    def toTorch(self, arr):
        torch_arr = arr.astype(np.float)
        torch_arr = torch_arr.reshape(1, *torch_arr.shape)
        return torch_arr    
    
    def run_epoch(self, sample_batch):
        """
        Runs an epoch and saves the checkpoint if there have been enough iterations

        :param sample_batch: A list containing inputs and labels

        """
                
        inputs = Variable(sample_batch[0]).float()
        labels = Variable(sample_batch[1]).float()

        inputs, labels = inputs.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()

        outputs = self.net(inputs)
        
        loss = self.criterion(torch.cat(outputs), labels.long())
        _, prediction = torch.max(torch.cat(outputs), 1)
        accuracy = torch.sum((prediction > 0) & (prediction==labels.long())).float()
        accuracy /= torch.sum((prediction > 0) | (labels > 0)).float()
        loss_hist = loss.cpu().item()
        loss.backward()
        self.optimizer.step()
        
        return loss_hist, accuracy.cpu().item()
    
    def evaluate(self, batch):       
        
        with torch.no_grad():
            inputs = Variable(batch[0]).float()
            labels = Variable(batch[1]).float()

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.net(inputs)
            
            loss = self.criterion(torch.cat(outputs), labels.long())
            _, prediction = torch.max(torch.cat(outputs), 1)
            accuracy = torch.sum((prediction > 0) & (prediction==labels.long())).float()
            accuracy /= torch.sum((prediction > 0) | (labels > 0)).float()
            
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
                self.save_checkpoint("best.ckpt")
        
        return loss.cpu().item(), accuracy.cpu().item(), torch.stack(outputs).cpu().numpy()
    
    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        num_iter = 1

        train_idx = np.random.permutation(len(self.inputs_volume[0]))
        train_idx = train_idx[:(len(train_idx) - len(train_idx) % 8)]
        train_idx = train_idx.reshape((-1, 8))
        val_idx = np.random.permutation(len(self.inputs_volume[1]))

        while num_epoch <= self.max_epochs:
            np.random.shuffle(train_idx)
            for i in range(train_idx.shape[0]):
                sample_batch = [np.stack([self.toTorch(self.inputs_volume[0][idx]) for idx in train_idx[i]]),
                                np.stack([self.labels_volume[0][idx] for idx in train_idx[i]])] 
                                
                if num_epoch > self.max_epochs:
                    break

                print("Iteration: {}".format(num_iter))
                train_loss, train_acc = self.run_epoch([torch.from_numpy(batch.astype(np.float)) for batch in sample_batch])
                if num_iter % self.checkpoint_period == 0:
                    self.save_checkpoint("iteration_{}.ckpt".format(num_iter))
                
                if num_iter % 10 == 0:
                    val_batch = [np.stack([self.toTorch(self.inputs_volume[1][idx]) for idx in val_idx[:16]]),
                                 np.stack([self.labels_volume[1][idx] for idx in val_idx[:16]])] 
                    loss, accuracy, _ = self.evaluate([torch.from_numpy(batch.astype(np.float)) for batch in val_batch])
                    self.logger.info("Iteration: {}, Epoch: {}/{}, Train loss: {:.4f}, Train acc: {:.2f}, Test loss: {:.4f}, Test acc: {:.2f}".format(num_iter, num_epoch, self.max_epochs, train_loss, train_acc*100, loss, accuracy*100))
                    
                num_iter += 1
        
            if num_epoch == self.max_epochs:
                while self.logger.handlers:
                    self.logger.handlers.pop()
            num_epoch += 1
    
    def save_checkpoint(self, checkpoint_name):
        """
        Saves a training checkpoint
        """
        checkpoint_filename = os.path.join(self.checkpoint_dir,
                                           checkpoint_name)
        torch.save(self.net.state_dict(), checkpoint_filename)