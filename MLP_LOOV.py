"""
Runs a fitting with PyTorch

This code is designed to handle datasets with a small number of observations.

Performs the fitting using Leave One Out Validation

Version 1.0.0
"""


import os
import torch
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm

# Custom module imports
from scripts.nets import *
from scripts.functions import *


# =========================================================================
# CODE USER PARAMETERS
# -------------------------------------------------------------------------
# Initial Data Source Parameters
OMIT    = ['SMILES', 'Median PR', 'CAS', 'Class'] # Which columns to ignore
TARG = 'Median PR'                                # Target Column
# Specify input file (.csv file containing the descriptors and labels)
INFILE  = "..\\For Azure - Mol2Vec\\data\\OPERA_Descriptors_N2.csv"
# -------------------------------------------------------------------------
# Suffix for output files - Note this may overwrite previous files
SUFF    = 'LOOV'
# -------------------------------------------------------------------------
# Neural Network Parameters
# -------------------------------------------------------------------------
LR     = 0.001 # Set the Learning Rate
LAYERS = 5  # Number of hidden Layers (1 to 5)
H1     = 8 # Number of nodes in hidden layer 1
H2     = 8 # Number of nodes in hidden layer 2
H3     = 8 # Number of nodes in hidden layer 3
H4     = 8 # Number of nodes in hidden layer 4
H5     = 8 # Number of nodes in hidden layer 5
DP     = 0. # Dropout probability
# -------------------------------------------------------------------------
# Splitting parameters for development and validation sets
TEST    = 0.0
DEV     = 0.1
# -------------------------------------------------------------------------
# Are there any features that need to be log transformed?
LOG  = (False, [])
OFFS = ('LogOffsets.pkl', 100) # offset reference and magnitude
# -------------------------------------------------------------------------
# Optimizer Parameters (Advanced parameters)
loss_fn      = nn.L1Loss()
AMSGRAD      = False
WEIGHT_DECAY = 0
num_epochs   = 20000
SAVE         = True
# =========================================================================


# Do not edit
try:
    os.mkdir('NN Files')
except OSError:
    pass
try:
    os.mkdir('NN_Models')
except OSError:
    pass
if TARG not in OMIT:
    OMIT.append(TARG)
# Identify device for fitting
use_cuda = torch.cuda.is_available()
device   = torch.device('cuda:0' if use_cuda else 'cpu')


def main():
    """main"""
    # Import and prepare the data

    # =========================================================================
    # Parameters to run
    # =========================================================================
    lr     = LR      # Set the Learning Rate
    layers = LAYERS  # Number of hidden Layers (1 to 5)
    h1     = H1      # Number of nodes in hidden layer 1
    h2     = H2      # Number of nodes in hidden layer 2
    h3     = H3      # Number of nodes in hidden layer 3
    h4     = H4      # Number of nodes in hidden layer 4
    h5     = H5      # Number of nodes in hidden layer 5
    dp     = DP      # Dropout probability
    # =========================================================================

    print('Predicting  :', TARG)
    print('Epochs      :', num_epochs)
    print('Save Results:', SAVE)
    print('Running     :', lr, h1, h2, h3, h4, dp, '\n')
    pkl.dump((lr, layers, h1, h2, h3, h4, h5, dp, AMSGRAD, WEIGHT_DECAY),
             open('NN Files\\%iLayer_NNHyperParameters_%s.pkl' % (LAYERS, SUFF), 'wb'))

    print("Importing Data:")
    cross = preprocess_data_xval(INFILE, TARG, dev=DEV, device=device,
                                 omit=OMIT, log=LOG, offs=OFFS)
    run   = 0

    print('\t>: Done.\n\nFitting NN:\n')
    for subset in tqdm(cross):
        K, TrainX, TrainY, DevX, DevY, TestX, TestY, scaler = subset
        best = fit_network(K, TrainX, TrainY, DevX, DevY, TestX, TestY,
                           lr=lr, hn1=h1, hn2=h2, hn3=h3, hn4=h4, hn5=h5,
                           dp_prob=dp, layers=layers, amsgrad=AMSGRAD,
                           weight_decay=WEIGHT_DECAY, save=SAVE, num_epochs=num_epochs,
                           loss_fn=loss_fn, suff=SUFF)

        pkl.dump((TrainX, TrainY, DevX, DevY, TestX, TestY, scaler),
                  open('NN Files\\DataSet_N%i_Set%i_%s.pkl' % (LAYERS, K, SUFF), 'wb'))
        state_out = open('NN Files\\NN_Completed_Run%i_%s.txt' % (run, SUFF), 'w')
        state_out.write('%s' % run)
        state_out.flush()
        state_out.close()
        run += 1

    print('\t>: Done.')


if __name__ in '__main__':
    main()
