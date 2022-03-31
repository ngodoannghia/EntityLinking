import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEmbedingLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden1, num_hidden2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, output_dim)

        self.lekyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
      
        x = self.fc1(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class MLPScoreLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden1, num_hidden2, num_hidden3, num_hidden4, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, num_hidden3)
        self.fc4 = nn.Linear(num_hidden3, num_hidden4)
        self.fc5 = nn.Linear(num_hidden4, output_dim)

        self.lekyrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):

        x = self.fc1(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.lekyrelu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x