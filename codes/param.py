import torch
import torch.optim as optim

s_seed = 1234
o_seed = 1234

# ------------ Define problem ------------
sdict = {
    'task' : 'var-one', # task name
    'v': 1.4, # inflating or shifting parameter
    'd': 16, # dimension
    'k': 2, # smoothness parameter (power of ReLU)
    'n' : {'P': 512, # number of datapoints for each P and Q
           'Q': 512}
}

# ------------ For grid search ------------
# Grid search only wors with d=2
gdict = {'w_cut': 200 + 1, # number of grids in [0, 2*\pi]. Two ends points are included.
         'b_bound': 5, # range of b is [0, b_bound].
         'b_cut': 100 + 1 # number of grids in [0, b_bound]. Two ends points are included.
}

# ------------ For optimization ------------
odict = {
    # Note 1. Total steps in optimization = tB * tI
    # Note 2. However, when 'verbose' is True, the optimization process prints
    #         the current result on every 'tB' steps. Thus, total prints are 'tI' times.
    #         Check more in tutorial.ipynb.
    #---
    'N' : 10, # number of neurons
    'lr' : 0.5, # learning rate
    'tB' : 100, # train_batch
    'tI': 12,   # train_iteration
    #---
    'lamb' : 1, # lambda
    'optimizer': 'ADAM', # type of optimizer
    'device': "cpu", # type of device
    'smpl_type': 'box_2' # sample type used in the calculation of test statistic
}