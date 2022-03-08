# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:03:36 2022

Contains all the additional tools needed to run the NNs

@author: BurnsT
"""


import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from glob import glob
from sklearn.utils import shuffle
from scipy.optimize import curve_fit as cfit
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.nets import *


def preprocess_data(layers=4, dev=0.1, test=0.0, device='cpu', suff=''):
    """preprocesses the data"""
    X, Y = import_data()

    # Create my subsets
    N, W = X.shape[0], X.shape[1]
    cut2 = int(dev*N)
    cut  = int((test + dev)*N)
    TrainX, TrainY = X[:-cut,:], Y[:-cut]
    DevX, DevY     = X[-cut:-cut2,:], Y[-cut:-cut2]
    TestX, TestY   = X[-cut2:,:], Y[-cut2:]
    X, Y = None, None

    # Scale my subsets
    scaler = StandardScaler()
    TrainX = scaler.fit_transform(TrainX)
    DevX   = scaler.transform(DevX)
    TestX  = scaler.transform(TestX)
    pkl.dump(scaler, open('NN Files\\Scaler_N%i_%s.pkl' % (layers, suff), 'wb'))
    print('Scaler_N%i.pkl' % (layers), 'written.')

    # Convert to and format Tensors
    TrainX = torch.from_numpy(TrainX).float()
    TrainX = TrainX.to(device)
    TrainY = TrainY.reshape(TrainY.shape[0], 1)
    TrainY = torch.from_numpy(TrainY).float()
    TrainY = TrainY.view(-1, 1)
    TrainY = TrainY.to(device)

    DevX = torch.from_numpy(DevX).float()
    DevX = DevX.to(device)
    DevY = DevY.reshape(DevY.shape[0], 1)
    DevY = torch.from_numpy(DevY).float()
    DevY = DevY.view(-1, 1)
    DevY = DevY.to(device)

    TestX = torch.from_numpy(TestX).float()
    TestX = TestX.to(device)
    TestY = TestY.reshape(TestY.shape[0], 1)
    TestY = torch.from_numpy(TestY).float()
    TestY = TestY.view(-1, 1)
    TestY = TestY.to(device)

    return TrainX, TrainY, DevX, DevY, TestX, TestY


def loov_sets(x, y):
    """Creates the LOOV sets"""
    x, y = randomize(x, y)
    train, valid = [], []
    
    for i in range(x.shape[0]):
        xsub = []
        ysub = []
        vx, vy = [], []
        for j in range(x.shape[0]):
            tx = [s for s in x[i,:]]
            ty = y[i]
            if i == j:
                vx.append(tx)
                vy.append(ty)
            else:
                xsub.append(tx)
                ysub.append(ty)
        xsub, ysub = np.array(xsub), np.array(ysub)
        vx, vy     = np.array(vx), np.array(vy)
        train.append((xsub, ysub, i))
        valid.append((vx, vy))
    return train, valid


def preprocess_data_xval(infile, targ, dev=0.1, device='cpu', 
                         omit=[], log=(False, []),
                         offs=('LogOffsets.pkl', 100)):
    """preprocesses the data"""
    X, Y = import_data(infile, targ, omit=omit, log=log)
    cross = []

    # Create my subsets
    train_sets, test_sets = loov_sets(X, Y)

    #TestX, TestY   = X[-cut2:,:], Y[-cut2:]
    X, Y,  = None, None

    for i, subset in enumerate(train_sets):
        x, y, k = subset
        N, W = x.shape[0], x.shape[1]
        cut = int(dev*N)

        vx, vy = test_sets[i]
        tx, ty = x[:-cut, :], y[:-cut]
        dx, dy = x[-cut:, :], y[-cut:]

        scaler = StandardScaler()
        tx = scaler.fit_transform(tx)
        dx = scaler.transform(dx)
        vx = scaler.transform(vx)

        tx = torch.from_numpy(tx).float()
        tx = tx.to(device)
        ty = ty.reshape(ty.shape[0], 1)
        ty = torch.from_numpy(ty).float()
        ty = ty.view(-1, 1)
        ty = ty.to(device)

        dx = torch.from_numpy(dx).float()
        dx = dx.to(device)
        dy = dy.reshape(dy.shape[0], 1)
        dy = torch.from_numpy(dy).float()
        dy = dy.view(-1, 1)
        dy = dy.to(device)

        vx = torch.from_numpy(vx).float()
        vx = vx.to(device)
        vy = vy.reshape(vy.shape[0], 1)
        vy = torch.from_numpy(vy).float()
        vy = vy.view(-1, 1)
        vy = vy.to(device)

        cross.append((i, tx, ty, dx, dy, vx, vy, scaler))

    return cross


def convergence_check(epochs, losses, check):
    """checks for convergence"""

    erange    = check # How many points will be considered
    converged = False

    def line(x, m=1, b=1):
        """line function"""
        return m*x+b

    # Set convergen threshold values
    #lm = -.05 # Low value of slope for it to be considered
    #hm = .04  # High value for slope
    lm = -.00001 # Low value of slope for it to be considered
    hm = .00001  # High value for slope

    # preprocess the data
    echeck = epochs[-erange:]
    scaler = StandardScaler()
    echeck = scaler.fit_transform([[i] for i in echeck])
    echeck = np.array([i[0] for i in echeck])
    lcheck = losses[-erange:]
    relval = np.mean(losses[-erange:]) # what value will be used for relative

    # Perform a linear fit to get an approximation of trend
    lout  = cfit(line, np.array(echeck),
                       np.array(lcheck/relval))
    m, b  = lout[0][0], lout[0][1]
    lX    = np.linspace(min(echeck), max(echeck), 500)
    lY    = line(lX, m=m, b=b)
    pY    = line(np.array([i for i in echeck]), m=m, b=b)
    stdval = np.std(losses[-erange:])

    # Get the relative values
    rm = 100* m / relval
    rstdval = 100* stdval / relval

    # Check for convergence
    if rm >= lm and rm <= hm:
        #print(m, rm)
        converged = True

    #return converged
    return False


def weights_init_uniform(m):
    """initializes the weights"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = np.sqrt(1/n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def import_data(infile, targ, omit=[], log=(False, []),
                offs=('LogOffsets.pkl', 100)):
    """imports the raw data"""
    raw = pd.read_csv(infile)

    cols = []
    for col in [c for c in raw]:
        if col in omit or col in targ:
            continue
        cols.append(col)
    cols.append(targ)
    if log[0]:
        offsets = {}
        for col in log[1]:
            msub   = raw.loc[raw[col] > 0]
            mini   = min(np.array(msub[col]))
            maxi   = max(np.array(raw[col]))
            offset = mini / offs[1]
            if min(np.array(raw[col])) > 0:
                disto = np.log10(np.array(raw[col]))
                offsets[col] = (False, offset)
            else:
                disto = np.log10(np.array(raw[col]) + offset)
                offsets[col] = (True, offset)
            raw[col] = disto
        pkl.dump(offsets, open(offs[0], 'wb'))
        print(offs[0], 'written.')
    rawd = raw[cols]
    raw  = None
    rawd = shuffle(rawd)
    data = np.array(rawd)
    rawd = None
    return data[:,:-1], data[:,-1]


def randomize(x, y):
    """randomizes the descr and label arrays"""
    I = [i for i in range(x.shape[0])]
    I = shuffle(I)
    x = np.array([x[i,:] for i in I])
    y = np.array([y[i] for i in I])
    return x, y


def fit_network(K, TrainX, TrainY, DevX, DevY, TestX, TestY, num_epochs=20000,
                lr=0.0001, layers=4, hn1=20, hn2=20, hn3=20,
                hn4=20, hn5=20, dp_prob=0., suff='',
                amsgrad=False, weight_decay=0, save=True, loss_fn=None):
    """Performs the NN fitting"""
    global use_cuda, device

    #print('>', lr, hn1, hn2, hn3, hn4, dp_prob)

    # Identify device for fitting
    use_cuda = torch.cuda.is_available()
    device   = torch.device('cuda:0' if use_cuda else 'cpu')
    #print('Device:', device)
    # Initialize the Network
    #net = Net(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
    #          nhidden3=hn3, nhidden4=hn4, dp_prob=dp_prob)
    if layers == 1:
        net = Net1(n_features=TrainX.shape[1], nhidden1=hn1, dp_prob=dp_prob)
    elif layers == 2:
        net = Net2(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
                   dp_prob=dp_prob)
    elif layers == 3:
        net = Net3(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
                   nhidden3=hn3, dp_prob=dp_prob)
    elif layers == 4:
        net = Net4(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
                   nhidden3=hn3, nhidden4=hn4, dp_prob=dp_prob)
    elif layers == 5:
        net = Net5(n_features=TrainX.shape[1], nhidden1=hn1, nhidden2=hn2,
                   nhidden3=hn3, nhidden4=hn4, nhidden5=hn5, dp_prob=dp_prob)
    net.apply(weights_init_uniform)
    net = net.to(device)

    # convergence parameters
    check     = 1000  # Check frequency and range
    minchk    = 5000  # mininum epochs to run before check
    converged = False

    # Run the Training
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, amsgrad=amsgrad, weight_decay=weight_decay)
    epX, epY, epR = [], [], []

    #print("Starting Fitting.")
    for epoch in tqdm(range(num_epochs), leave=False):
        #print('Training on Epoch %i' % epoch)
        train_loss, valid_loss = 0, 0

        net.train()
        optimizer.zero_grad()

        # Forward Propagation
        y_train_predict = net(TrainX)

        # Loss Function
        loss = loss_fn(y_train_predict, TrainY)

        # Backward Propagation
        loss.backward()

        # Weight Opt
        optimizer.step()
        train_loss = float(loss) #.detatch())

        # Evaludate the fitting
        net.eval()
        y_dev_predict = net(DevX)
        DevY = DevY.to(device)
        loss = loss_fn(y_dev_predict, DevY)
        valid_loss = float(loss) #.detatch())
        #print('\tDev Loss =', valid_loss)
        epX.append(epoch)
        epY.append(valid_loss)
        #if epoch > minchk and epoch % check == 0:
        #    #print(epoch, check, epoch%check)
        #    converged = convergence_check(epX, epY, check)
        #if converged:
        #    break
    #print("\tDone")

    if save:
        net.eval()
        y_test_pred = net(TestX).detach()
        TestY = TestY.cpu().numpy()
        y_test_pred = y_test_pred.cpu().numpy()
        pkl.dump((epX, epY, epR, y_test_pred, TestY), open('NN Files\\NewResults_Set%i_%s.pkl' % (K, suff), 'wb'))
        model_ini = 'NN_Models\\NN_Model_%iLayers_Targ_Set%i' % (layers, K)
        model_file = model_ini + '_%i.pkl' % len(glob('%s_*.pkl' % model_ini))
        torch.save(net.state_dict(), model_file)

    return min(epY)
