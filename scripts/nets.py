# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:30:21 2022

This is a script that contains all of the MLP net objects

@author: BurnsT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()
device   = torch.device('cuda:0' if use_cuda else 'cpu')


class Net1(nn.Module):
    """
    Neural Net Class
    
    1 Hidden Layer
    """

    def __init__(self, n_features=20, nhidden1=20, act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.output  = nn.Linear(nhidden1,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x


class Net2(nn.Module):
    """
    Neural Net Class
    
    2 Hidden Layers
    """

    def __init__(self, n_features=20, nhidden1=20, nhidden2=20,
                 act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.nhidden2   = nhidden2
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.fc2     = nn.Linear(nhidden1,   nhidden2)
        self.output  = nn.Linear(nhidden2,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.act_fn(self.fc2(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x


class Net3(nn.Module):
    """
    Neural Net Class
    
    3 Hidden Layers
    """

    def __init__(self, n_features=20, nhidden1=20, nhidden2=20,
                 nhidden3=20, act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.nhidden2   = nhidden2
        self.nhidden3   = nhidden3
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.fc2     = nn.Linear(nhidden1,   nhidden2)
        self.fc3     = nn.Linear(nhidden2,   nhidden3)
        self.output  = nn.Linear(nhidden3,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.act_fn(self.fc2(x)))
        x = self.dropout(self.act_fn(self.fc3(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x



class Net4(nn.Module):
    """
    Neural Net Class
    
    4 Hidden Layers
    """

    def __init__(self, n_features=20, nhidden1=20, nhidden2=20,
                 nhidden3=20, nhidden4=20, act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.nhidden2   = nhidden2
        self.nhidden3   = nhidden3
        self.nhidden4   = nhidden4
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.fc2     = nn.Linear(nhidden1,   nhidden2)
        self.fc3     = nn.Linear(nhidden2,   nhidden3)
        self.fc4     = nn.Linear(nhidden3,   nhidden4)
        self.output  = nn.Linear(nhidden4,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.act_fn(self.fc2(x)))
        x = self.dropout(self.act_fn(self.fc3(x)))
        x = self.dropout(self.act_fn(self.fc4(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x


class Net5(nn.Module):
    """
    Neural Net Class
    
    5 Hidden Layers
    """

    def __init__(self, n_features=20, nhidden1=20, nhidden2=20,
                 nhidden3=20, nhidden4=20, nhidden5=20,
                 act_fn=F.relu, dp_prob=0.1):
        super().__init__()
        # Initializa all net specific parameters
        self.n_features = n_features
        self.nhidden1   = nhidden1
        self.nhidden2   = nhidden2
        self.nhidden3   = nhidden3
        self.nhidden4   = nhidden4
        self.nhidden5   = nhidden5
        self.act_fn     = act_fn   # Define the non-linear activation function
        self.dp_prob    = dp_prob  # Dropout layer probability

        # Initialize Layers of the Net:
        # Note:  First Fully Connected Layer (fc1) where nn.Learnear has (X, Y)
        #        and X = number of features fed into payer, Y = number of features returned
        #        by layer (X=input, Y=output)
        self.fc1     = nn.Linear(n_features, nhidden1)
        self.fc2     = nn.Linear(nhidden1,   nhidden2)
        self.fc3     = nn.Linear(nhidden2,   nhidden3)
        self.fc4     = nn.Linear(nhidden3,   nhidden4)
        self.fc5     = nn.Linear(nhidden4,   nhidden5)
        self.output  = nn.Linear(nhidden5,   1)
        self.dropout = nn.Dropout(p=dp_prob)

    def forward(self, x):
        """forward propagation in the NN"""
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.act_fn(self.fc2(x)))
        x = self.dropout(self.act_fn(self.fc3(x)))
        x = self.dropout(self.act_fn(self.fc4(x)))
        x = self.dropout(self.act_fn(self.fc5(x)))
        x = self.act_fn(F.relu(self.output(x)))
        return x
