import os, sys
from IPython.display import display, HTML

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#----------------------------------------
def print_basic_info():
    """
    Print brief information about the environment.
    """
    
    CURDIR = os.getcwd(); print(f"{'Current directory' :<22}", CURDIR)
    print(f"{'Python' :<22}", sys.version.replace("\n", ""))
    print(f"{'Pytorch' :<22}", torch.__version__)
    print(f"{'Pandas' :<22}", pd.__version__)
    print(f"{'MPS available':<22}", torch.backends.mps.is_available(),) # this ensures that the current MacOS version is at least 12.3+
    print(f"{'MPS built':<22}", torch.backends.mps.is_built()) # this ensures that the current PyTorch installation was built with MPS activated.
    print(f"{'Set IPython display' :<22} Done")
    display(HTML("<style>.container { width:100% !important; }</style>"))
    display(HTML("<style>.output_result { max-width:100% !important; }</style>"))

#----------------------------------------
def PlotAltNullRepeat(mysample, output, log_nolog:str = None, eff_tI:int = None):
    """
    Plot the histograms and the ROC curves for different tests, based on the output of "sampling.MySample.AltNullRepeat".
    The tests visualized here are: RKS test, oracle test, kernel MMD test (linear, quadratic, cubic, gaussian), and energy distance test.
    
    Input:
    - mysample (sampling.Mysample): Includes all setup for the task and the dataset.
    - output: The output of of "mysample.AltNullRepeat".
    - log_nolog (str): The type of the optimization objective, whose result is plotted in this function.
    - eff_tI (int): Indicates the 'effective' iteration in the optimization process. 'Effective' means the result when the optimization processed stopped at 'eff_tI * mysample.tB', not 'mysample.tAll'.
    """
    
    # Basic setup
    eps = 1e-32
    
    if log_nolog is None: log_nolog = 'log'
    if eff_tI is None: eff_tI = mysample.tI
    assert log_nolog in ['log', 'nolog']
    
    print(f"{log_nolog} / {eff_tI}")

    temp = output[output['log_nolog'] == log_nolog]
    temp = temp[temp['eff_tI'] == eff_tI]

    # Get data to visualize
    for value in ['ipm_max', 'oracle', 'kmmd_1', 'kmmd_2', 'kmmd_3', 'kmmd_g', 'energy']:
        alt  = temp[temp['hypo'] == 'alt_hypo'][value].values.astype(float)
        null = temp[temp['hypo'] == 'null_hypo'][value].values.astype(float)
        assert alt.shape == null.shape

        # Plot histogram
        xmin = np.min([alt, null])
        xmax = np.max([alt, null])
        fig, axes = plt.subplots(1, 2, figsize = (7, 3.5))
        axes[0].hist(alt, range = (xmin, xmax), bins=100, alpha=0.6, label = "alt_hypo")
        axes[0].hist(null, range = (xmin, xmax),  bins=100, alpha=0.6, label = "null_hypo")
        axes[0].legend()

        if value in ['ipm_max', 'ipm_best']:
            title = f"{value} (k={mysample.k})"
        else:
            title = value
        axes[0].set_title(f"{title} test statistics hist.")

        # Plot ROC curves
        cuts = 5000
        visdata = {'null': np.zeros(cuts), 'alt': np.zeros(cuts)}
        t_list = np.linspace(xmin, xmax, cuts).tolist()   
        for (i, t) in enumerate(t_list):
            visdata['null'][i] = np.mean(null > t)
            visdata['alt'][i]  = np.mean(alt > t)

        true_pos = visdata['alt']
        false_pos = visdata['null']

        axes[1].plot(false_pos, true_pos)
        axes[1].set_title(f"{title} test statistics ROC.")

        fig.tight_layout()
        fig.show()

        
#----------------------------------------
def heatmap_function(x, y, k, gs_output):
    """
    Return the value of ReLU^k at (x,y) in 2-dimension. 

    Input:
    - x, y: the output of numpy.meshgrid of two numpy arrays
    - k (float): smoothness parameter
    - gs_output (dict): output of "sampling.MySample.GridSearch"
    """
    gs_k = gs_output[k]['grid_best']
    w = gs_k['w'].numpy()
    b = gs_k['b']
    if k > 0:
        lin_relu = np.maximum(w[0] * x + w[1] * y - b, 0)
        return np.power(lin_relu, k)
    else:
        zero_one = (w[0] * x + w[1] * y - b) > 0
        return zero_one.astype(int)

#----------------------------------------
def DrawHeatmaps(k_list, mysample, gdict):
    """
    Draw the scatter plot of the samples, and multiple heatmaps of ReLU^k with different values of k, such that maximize the MMD value of the sample.
    This function must be called after "mysample.GenSmpl()" is successfully called to generate samples.
    
    Input:
    - klist (list): the values of k to draw the heatmaps.
    - mysample (sampling.Mysample): Includes all setup for the task and the dataset. Moreover, since "mysample.GenSmpl()" must be called before running this function, it already includes the sampled dataset.
    - gdict (dict): Defined in param.py. Includes the resolution and the range for the grid search.
    """
    
    if mysample.d != 2:
        print("Grid search only works for d=2 so far.")
        return
    
    # Do grid search for each value of k, and save result in 'gs_output'
    gs_output = {k: None for k in k_list}
    
    print("Grid search:", end =" ")
    for k in k_list:
        print(k, end=" ")
        mysample.k = k
        mysample.verbose = False
        mysample.GridSearch(gdict)
        gs_output[k] = {'grid_hist': mysample.grid_hist.copy(), 'grid_best': mysample.grid_best.copy()}
       
    # --------- Draw plots: Start ---------- #
    fig = plt.figure(figsize=(4 * len(k_list) + 1, 4))
    gs = gridspec.GridSpec(1, 1+len(k_list))

    xdata, ydata = mysample.smpl_cent['All'][:, 0].numpy(), mysample.smpl_cent['All'][:, 1].numpy()
    xmin, xmax = np.min(xdata), np.max(xdata)
    ymin, ymax = np.min(ydata), np.max(ydata)
    allmin, allmax = 1.05 * np.min([xmin, ymin]), 1.05 * np.max([xmax, ymax])
    
    ### 1. Draw scatter plots of datapoints
    ax = fig.add_subplot(gs[0])
    color = {'P': 'b', 'Q': 'r'}
    alpha = {'P': 0.3, 'Q': 0.6}
    for pq in ['P', 'Q']:
        col, alp = color[pq], alpha[pq]
        sPQ = mysample.smpl_cent[pq]
        ax.scatter(sPQ[:, 0], sPQ[:, 1], s=3, c=col, alpha=alp, label=pq)
        ax.set_xlim(allmin, allmax)
        ax.set_ylim(allmin, allmax)
        ax.legend(fontsize = 15)
        
    ### 2. Draw heatmaps
    x, y = np.linspace(allmin, allmax, 400), np.linspace(allmin, allmax, 400)
    X, Y = np.meshgrid(x, y)
    Z = {k: heatmap_function(X, Y, k, gs_output) for k in k_list}

    # 2.1 Draw heatmaps for each k
    print("   Heatmap:", end =" ")
    for i in range(len(k_list)):
        k = k_list[i]
        print(k, end=" ")
        ax = fig.add_subplot(gs[1+i])
        im = ax.imshow(Z[k], origin='lower', cmap='hot', aspect='auto', extent=[allmin, allmax, allmin, allmax])
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=-0.05)  # Adjust colorbar position

        if k > 0:
            zmin, zmax = np.floor(Z[k].min()).astype(int), np.ceil(Z[k].max()).astype(int)
            levels = list(range(0, zmax+1, 3))
            contour = ax.contour(X, Y, Z[k], levels=levels, colors='grey', extent=[allmin, allmax, allmin, allmax])

        ax.set_title(f"k={k}", fontsize=15)
        # cbar.set_label('Function Value')

    # 3 Adjust subplot layout manually
    fig.subplots_adjust(left=0.1, right=1.2, bottom=0.1, top=0.9, wspace=0.3)

    # Show and save the plot
    plt.show()
    fig.savefig('temp.pdf', bbox_inches='tight', pad_inches=0)
        
    