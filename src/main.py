import argparse
import sys

import torch
from torch import nn
from torch import optim

from data import mnist
from model import MyAwesomeModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        args = parser.parse_args(sys.argv[2:])
        print(args)
        model = MyAwesomeModel()
        train_set, test_set = mnist()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters())
        epochs = 30
        train_losses, test_losses = [], []
        for e in range(epochs):
            model.train()
            running_loss = 0
            for images, labels in train_set:
                optimizer.zero_grad()
                images = images.view(images.shape[0], -1)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            running_loss = running_loss / len(train_set)
            with torch.no_grad():
                model.eval()
                accuracy = 0
                val_loss = 0
                for val_images, val_labels in test_set:
                    val_images = val_images.view(val_images.shape[0], -1)
                    val_log_ps = model(val_images)
                    val_loss += criterion(val_log_ps, val_labels)
                    val_ps = torch.exp(val_log_ps)
                    top_p, top_class = val_ps.topk(1, dim=1)
                    matches = top_class == val_labels.view(top_class.shape)
                    accuracy += torch.mean(matches.type(torch.FloatTensor))
                val_loss = val_loss / len(test_set)
                accuracy = accuracy / len(test_set)
            print(f"Training loss: {running_loss} Validation loss: {val_loss} Accuracy: {accuracy*100}%")
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        model = torch.load(args.load_model_from)
        _, test_set = mnist()
        with torch.no_grad():
            model.eval()
            accuracy = 0
            for val_images, val_labels in test_set:
                val_images = val_images.view(val_images.shape[0], -1)
                val_log_ps = model(val_images)
                val_ps = torch.exp(val_log_ps)
                top_p, top_class = val_ps.topk(1, dim=1)
                matches = top_class == val_labels.view(top_class.shape)
                accuracy += torch.mean(matches.type(torch.FloatTensor))
            accuracy = accuracy / len(test_set)
        print(f"Accuracy: {accuracy*100}%")
if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    