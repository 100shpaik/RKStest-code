########## Import packages ##########
import random, time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

from sklearn import metrics
from sklearn import linear_model
from scipy.stats import t

import dcor # This package is for Energy Distance test.
            # (Check "MySample.RivalTestStatistic" method to see how it is used.)

########## For kernel MMD ##########
def MMD_calculator(X, Y, kernel:str, degree_poly=2, gamma_poly=1.0, gamma_gaus=1.0, coef0=0.0):
    assert kernel in ['linear', 'polynomial', 'gaussian', 'gaussian-est']
    
    if kernel == 'linear':
        XX = metrics.pairwise.polynomial_kernel(X, X, degree = 1, gamma = 1, coef0 = 0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree = 1, gamma = 1, coef0 = 0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree = 1, gamma = 1, coef0 = 0)

    elif kernel == 'polynomial':
        XX = metrics.pairwise.polynomial_kernel(X, X, degree_poly, gamma_poly, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree_poly, gamma_poly, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree_poly, gamma_poly, coef0)
        
    elif kernel == 'gaussian-est':
        X_and_Y = torch.cat((X, Y), dim=0)
        sig_est = np.median(metrics.pairwise.euclidean_distances(X_and_Y, X_and_Y))
        gamma_est = 1/(2 * sig_est **2)
        XX = metrics.pairwise.rbf_kernel(X, X, gamma_est)
        YY = metrics.pairwise.rbf_kernel(Y, Y, gamma_est)
        XY = metrics.pairwise.rbf_kernel(X, Y, gamma_est)
        
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

########## For k=0: this is only used in MySample.IpmGivenWB, and not used in main experiment
class LR_IpmCalculation(nn.Module):
    def __init__(self, d):
        super(LR_IpmCalculation, self).__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        x = self.lin(x)
        return x

########## For k>=0: this single class includes everything we need ##########
class MySample():
    #----------------------------------------
    def __init__(self, sdict, s_seed = None, verbose = False):
        """
        Input:
        - sdict (dict): Defined in "param.py" file. This includes the information for sampling.
        """
        
        assert sdict is not None

        self.verbose = verbose # :see more for the method called "PrintIf" in this class

        if s_seed is None:
            self.s_seed = 2023
        else:
            self.s_seed = s_seed

        ### initialization for sampling
        self.d, self.k,  = sdict['d'], sdict['k']
        self.n = sdict['n'] 
        self.n['All'] = self.n['P'] + self.n['Q']
        self.task = sdict['task']
        self.v = sdict['v']
        self.null_count = {'P': 0, 'Q': 0}

        self.dist = {'P': None, 'Q': None} # distributions
        self.smpl = {'P': None, 'Q': None, 'All': None} # samples 
        self.smpl_cent = {'P': None, 'Q': None, 'All': None} # centered samples
        self.smpl_avg = {'P': None, 'Q': None, 'All': None} # sample mean
        self.smpl_var = {'P': None, 'Q': None, 'All': None} # sample variance

        self.smpl_box_2 = {'P': None, 'Q': None, 'All': None}

        ### initialization for grid search    
        self.w_cut, self.b_bound, self.b_cut = None, None, None
        self.gird_hist = None
        self.grid_best = {'IPM': float("nan")} # :see "GridSearch(self, gdict)" for more

        ### initialization for permutation
        #self.permute_smpl = {'P': None, 'Q': None, 'All': None}
        #self.permute_ipm = {'permute_best':[], 'permute_last':[]}

        ### initialization for repetition
        self.repetition = None
    
    #----------------------------------------
    def PrintIf(self, txt:str, end='\n'):
        """
        Print something when self.verbose is True, and don't print otherwise
        """
        if self.verbose: print(txt, end=end)
    
    #----------------------------------------
    def GenDist(self):
        """
        Generate distributions, P and Q. The actual sampling does not happen in this method.
        """
        
        assert self.task in ['var-one', 'var-all', 'mean-shift', 'kmmd-iso', 't-one']

        if self.task == 'var-one':
            self.used_v = self.v
            covQ = torch.eye(self.d)
            covQ[0, 0] = self.used_v 
            self.dist['P'] = MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)) # Normal (0, I_d)
            self.dist['Q'] = MultivariateNormal(torch.zeros(self.d), covQ) # Normal (0, diag(v, 1, ..., 1))

        elif self.task == 'var-all':
            self.used_v = self.v
            covQ = torch.eye(self.d) * self.used_v
            self.dist['P'] = MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)) # Normal (0, I_d)
            self.dist['Q'] = MultivariateNormal(torch.zeros(self.d), covQ) # Normal (0, v * I_d)

        elif self.task == 'mean-shift':
            self.used_v = self.v / 8
            meanQ = torch.zeros(self.d)
            meanQ[0] = self.used_v
            self.dist['P'] = MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)) # Normal (0, I_d)
            self.dist['Q'] = MultivariateNormal(meanQ, torch.eye(self.d)) # Normal ([v/8, 0, ..., 0], I_d)

        elif self.task == 'kmmd-iso':
            self.used_v = self.v / 8
            meanQ = torch.zeros(self.d)
            meanQ[0] = self.used_v
            cov = torch.eye(self.d) * (3**2)
            cov[0,0] = 1
            self.dist['P'] = MultivariateNormal(torch.zeros(self.d), cov) # Normal (0, diag(1, 9, ... 9))
            self.dist['Q'] = MultivariateNormal(meanQ, cov) # Normal (v/8, diag(1, 9, ... 9))

        elif self.task == 't-one':
            self.used_v = self.v
            self.dist['P'] = MultivariateNormal(torch.zeros(self.d), torch.eye(self.d)) # Normal (0, I_d)
            self.dist['Q'] = {'t'      : torch.distributions.studentT.StudentT(self.v, loc=0, scale=1.0),  
                               'normal': MultivariateNormal(torch.zeros(self.d-1), torch.eye(self.d-1))  }

        self.PrintIf(f"P: {self.dist['P']}\nQ: {self.dist['Q']}")

    
    #----------------------------------------
    def GenSmpl(self, HYPO = 'alt', rep_seed = 0): 
        """
        Sample data in P and Q. This method must be called after "GenDist" method.
        
        Input:
        - HYPO (str): Rerpesents the alternative or null hypothesis
        - rep_seed (int): When we resample the data multiple times, this is used not to resample the same data.
        """
        
        assert HYPO in ['alt', 'null']
        self.hypo = HYPO
        self.null_count = {'P': 0, 'Q': 0}
        self.PrintIf(f'Sampling starts ({HYPO :<4})', end =' ')
        start = time.time()
        
        seed = self.s_seed + rep_seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        for pq in ['P', 'Q']:
            non_pq = 'PQ'.replace(pq, '')
            nPQ = self.n[pq]
            samples = torch.zeros((nPQ, self.d), dtype = torch.float)
            
            ### Alternative hypothesis
            if HYPO == 'alt':
                dist, type_dist = self.dist[pq], type(self.dist[pq])
                        
                if type_dist == torch.distributions.multivariate_normal.MultivariateNormal:
                    for i in range(nPQ):
                        row = dist.sample().to(dtype = torch.float)
                        samples[i] = row
                        
                elif type_dist == dict:
                    if self.task == 't-one':
                        for i in range(nPQ):
                            row1 = dist['t'].sample().to(dtype = torch.float)
                            row2 = dist['normal'].sample().to(dtype = torch.float)
                            row = torch.cat((row1.unsqueeze(0), row2))
                            samples[i] = row
                else:
                    print("\nUnimplemented distribution for sampling!")
                    return None
                
            ### Null hypothesis
            elif HYPO == 'null':
                for i in range(nPQ):
                    # Choose P or Q: based on nP and nQ
                    unif = np.random.uniform(low = 0, high = 1)
                    if unif < self.n['P'] / self.n['All']:
                        dist = self.dist["P"]
                    else:
                        dist = self.dist["Q"]
                    type_dist = type(dist)
                    
                    # Sampling is the same as the alt. hypo case
                    if type_dist == torch.distributions.multivariate_normal.MultivariateNormal:
                        row = dist.sample().to(dtype = torch.float)
                        samples[i] = row
                        
                    elif type_dist == dict:
                        if self.task == 't-one':
                            row1 = dist['t'].sample().to(dtype = torch.float)
                            row2 = dist['normal'].sample().to(dtype = torch.float)
                            row = torch.cat((row1.unsqueeze(0), row2))
                            samples[i] = row
                    else:
                        print("\nUnimplemented distribution for sampling!")
                        return None
            else:
                print("\nHypothesis should be alt (default) or null!")
                return None
        
            self.smpl[pq] = samples
        ### --- #nd of "for pq in ['P', 'Q']" loop --- ###

        self.PrintIf(f"Sampling ends. Time = {round(time.time() - start, 4)} sec")
        self.smpl['All'] = torch.cat((self.smpl['P'], self.smpl['Q']), dim = 0)

        # Get the sample mean and variance
        for pq in ['P', 'Q', 'All']:
            self.smpl_avg[pq] = torch.mean(self.smpl[pq], dim=0)
            self.smpl_var[pq] = torch.var(self.smpl[pq], dim=0)

        # Center the samples
        for pq in ['P', 'Q', 'All']:
            self.smpl_cent[pq] = self.smpl[pq] - self.smpl_avg['All']
            
        # Scale the samples with L2 norm
        self.mean_L2 = torch.mean(torch.sum(torch.pow(self.smpl_cent['All'], 2), dim=1)) ** 0.5
        for pq in ['P', 'Q', 'All']:
            self.smpl_box_2[pq] = self.smpl_cent[pq] / self.mean_L2

    #----------------------------------------
    def SmplSummary(self):
        """
        Print the summary of the sample. This should be called after "GenSmpl" method.
        """
        
        print(f"s_seed = {self.s_seed}")
        print(f"d = {self.d} / k = {self.k} / task = {self.task} / v = {self.v}")
        print(f"nP= {self.n['P']} / nQ = {self.n['Q']}")
        print("hypothesis =", self.hypo)
        for pq in ['P', 'Q', 'All']:
            avg, var = torch.round(self.smpl_avg[pq], decimals = 4), torch.round(self.smpl_var[pq], decimals = 4)
            print(pq, ": avg =", avg, "/ var =", var)
    
    #----------------------------------------
    def VisDimOneTwo(self, smpl_type = None, color:dict = {'P': 'b', 'Q': 'r'}, alpha:dict ={'P': 0.3, 'Q': 0.6}):
        """
        2D scatter plot of the sample. This should be called after "GenSmpl" method.
        When self.d is larger than 2, only the first and the second dimensions are chosen to be plotted.
        """
        
        assert smpl_type is not None
        plt.figure(figsize = (3, 3))
        for pq in ['Q', 'P']:
            col, alp = color[pq], alpha[pq]
            if smpl_type == 'original':
                sPQ = self.smpl[pq]
            elif smpl_type == 'centered':
                sPQ = self.smpl_cent[pq]
            elif smpl_type == 'box_2':
                sPQ = self.smpl_box_2[pq]
                
            plt.scatter(sPQ[:, 0], sPQ[:, 1], s=2, c=col, alpha=alp, label=pq)
                
        plt.axis('equal')
        plt.legend()

    #----------------------------------------
    def IpmGivenWB(self, w, b, a, permute = False, mult_neurons = False):
        """
        Calculate the MMD (or IPM - integral probability metric) value of the samples, with given w, b, and a.
        
        Input:
        - w: (d, N)-dimensiona vector, representing N directions in R^d. N is the number of the neurons.
        - b: (1, N)-dim vector, representing the the offset in R^d. (See the explanation of Radon Transfrom.)
        - a: N-dimensional vector, representing the mass (or the measure) for N neurons.
        - mult_neurons: Whether the multiple neurons are used or not.
        """
        
        N = self.N if mult_neurons else 1
        
        ### --- k=0: Start --- ###
        ### Note that 'IpmGivenWB' method with k=0 is only called in grid search, and not used in main experiment.
        if self.k == 0:
            nP, nQ = self.n['P'], self.n['Q']
            model = LR_IpmCalculation(self.d)
            model.lin.weight.data, model.lin.bias.data = w, b
            model.eval()
            
            y = torch.cat([torch.zeros(nP), torch.ones(nQ)], dim=0).unsqueeze(1)
            if not permute:
                if self.smpl_type == 'original':
                    X = self.smpl['All']
                elif self.smpl_type == 'centered':
                    X = self.smpl_cent['All']
                elif self.smpl_type == 'box_2':
                    X = self.smpl_box_2['All']
            else:
                X = self.permute_smpl['All']

            pred = model(X)
            pred = torch.sigmoid(pred) >= 0.5
            pred = y[pred]
            countP, countQ = torch.sum(pred < 0.5), torch.sum(pred >= 0.5)
            
            ipm = torch.abs(countP/nP - countQ/nQ).item()
            
            return ipm, None
        ### --- End of k=0 --- ###
        
        
        if self.smpl_type == 'original':
            sP, sQ = self.smpl['P'], self.smpl['Q']
        elif self.smpl_type == 'centered':
            sP, sQ = self.smpl_cent['P'], self.smpl_cent['Q']
        elif self.smpl_type == 'box_2':
            sP, sQ = self.smpl_box_2['P'], self.smpl_box_2['Q']

        with torch.no_grad():
            relu, lin = nn.ReLU(), nn.Linear(self.d, N)
            lin.weight, lin.bias = nn.Parameter(w), nn.Parameter(b)

            yP  = torch.sum(torch.pow(relu(lin(sP)), self.k), dim = 0) / self.n['P']
            yQ  = torch.sum(torch.pow(relu(lin(sQ)), self.k), dim = 0) / (-self.n['Q'])
            y   = yP + yQ
            ipm = torch.sum(torch.mul(y, a))

        return ipm.item(), lin
    
    #----------------------------------------
    def GridSearch(self, gdict):
        """
        Find MMD value by grid search. Only available for 2 dimension.
        
        Input:
        - gdict (dict): Defined in param.py. Includes the resolution and the range for the grid search.
        """
        
        if self.d > 2:
            print("Grid search only works for d=2 so far.")
            return None
        
        ### Initialization for grid search
        self.w_cut, self.b_bound, self.b_cut = gdict['w_cut'], gdict['b_bound'], gdict['b_cut']
        self.grid_hist = {'IPM':[float("-inf") for i in range(self.w_cut)],
                          'b':[float("inf") for i in range(self.w_cut)],
                          'total_time': 0}
        self.grid_best = {'IPM': float("-inf"),
                          'i': float("-inf"),
                          'w': float("-inf"),
                          'b': float("-inf")}
        theta = np.linspace(0, 2*np.pi, self.w_cut)
        w_cand = torch.stack( (torch.Tensor(np.cos(theta)), torch.Tensor(np.sin(theta))) , dim=1)
        b_cand = torch.Tensor(np.linspace(0, self.b_bound, self.b_cut))
        
        ### --- Grid search begins --- ###
        start = time.time()
        self.PrintIf(f"task: {self.task: <10} / v: {self.v: <2} / k = {self.k: <2}: Grid.", end = " ")
        for i in range(self.w_cut):
            w = w_cand[i].unsqueeze(dim = 0) # unsqueeze is for nn.Linear(d,1).weight
            max_ipm= float("-inf")
            for b in b_cand:
                ipm, _ = self.IpmGivenWB(w, -b, torch.Tensor([1.])) 
                ipm = abs(ipm)
                
                if self.smpl_type == 'box_2':
                    ipm = ipm * (self.mean_L2.item() ** self.k) 
                    
                if max_ipm < ipm:
                    max_ipm, max_b = ipm, b.item()
            
            ### Save history
            self.grid_hist['IPM'][i], self.grid_hist['b'][i] = max_ipm, max_b

            ### Update the best
            if self.grid_best['IPM'] < max_ipm:
                self.grid_best['IPM'] = max_ipm
                self.grid_best['i']   = i
                self.grid_best['w']   = w.squeeze() 
                self.grid_best['b']   = max_b

        ### --- Grid search ends --- ###
        
        ### Save total time
        self.grid_hist['total_time'] = round(time.time()-start, 3)
        
        ### Print results
        ptime = self.grid_hist['total_time']
        pipm = round(self.grid_best['IPM'], 4)
        pw = torch.round(self.grid_best['w'], decimals = 4)
        pb = round(self.grid_best['b'], 4)
        txt = f"End. Time = {ptime: <6} sec / IPM = {pipm: <8} / w = {pw} / b = {pb}"
        self.PrintIf(txt)
    
    #----------------------------------------
    def VisGridSearch(self, log_plot = False):
        """
        Visualize the result of the grid search. This method should be called after "GridSearch" method.
        
        Input:
        - log_plot: Whether change the MMD value from the grid search into the log scale or not. This is a included since the (regularized) MMD optimization uses the log(MMD) value.
        """
        
        if self.d !=2:
            print("Grid search only works for d=2 so far.")
            return None
        if self.grid_hist is None or self.grid_best is None:
            print("Do the grid search first (in case d=2): use 'GridSearch'.")
            return None
        
        y_data = self.grid_hist['IPM']
        
        ### For log_plot: data
        if log_plot:
            y_data = np.log(y_data)
        
        ### For attributes of plot
        max_idx = np.argmax(y_data)
        max_y = round(np.max(y_data), 4)
        txt1 = f"w = {self.grid_best['w']}, b = {self.grid_best['b']}"
        txt2 = f"nP = {self.n['P']}, nQ = {self.n['Q']}, b in [{0}, {self.b_bound}], hypo = {self.hypo}"

        ### Draw plot
        plt.figure(figsize = (4,4))
        plt.plot(range(self.w_cut), y_data)
        plt.axvline(x=max_idx, color='r', linewidth = 1, label = 'w: ' +str(max_idx))
        plt.title(f"Grid: k = {self.k} / d = {self.d} / task = {self.task} / v= {self.v}")
        plt.xlabel("w (0 to 2pi)")
        plt.figtext(0.5, -0.08, txt1 + "\n" + txt2, wrap=True, horizontalalignment='center', fontsize=10);
        
        ### For log_plot: attribute
        if log_plot:
            plt.axhline(y=max_y, color='r', linewidth = 1, label = 'log(IPM): ' +str(max_y))
            plt.ylabel("log(IPM)")
        else:
            plt.axhline(y=max_y, color='r', linewidth = 1, label = 'IPM: ' +str(max_y))
            plt.ylabel("positive IPM")
            
        plt.legend(loc = 'upper left')

    #----------------------------------------
    def OptInit(self, odict, o_seed = None):
        """
        Input:
        - odict (dict): Defined in the "param.py" file. This includes the information for optimization.
        - o_seed (int): The seed for reproducibility.
        """
        
        if o_seed is None:
            self.o_seed = 1210
        else:
            self.o_seed = o_seed
            
        self.N = odict['N']
        self.lr = odict['lr']
        self.tB, self.tI = odict['tB'], odict['tI']
        self.tAll = self.tB * self.tI
        
        self.lamb = odict['lamb'] # lambda
        self.optimizer = odict['optimizer']
        self.device = odict['device']
        self.smpl_type = odict['smpl_type']
        
    #----------------------------------------
    def OptSolve(self, rep_seed = 0, use_log = True, checkpoints = []):
        """
        Solve the optimiazation problem of the form 'log(MMD) + penalty' as explained in the paper.
        
        Input:
        - rep_seed (int): When the same optimization problem is solved several times, to change the initialization of the parameter.
        - use_log (bool): When it is true, solve 'log(MMD) + penalty'. When it is False, solve 'MMD + penalty'
        - checkpoints (list): In each 'checkpoint (int)', save the output. This can be an empty list.
        """

        ### Initialize history of optimization
        self.opt_hist = []
        self.opt_nuisance = {'y':[], 'penalty':[]}
        
        ### For checkpoint: initialization
        if len(checkpoints) > 0:
            self.opt_checkpoint = {'best':{}, 'best_one':{}}
            
        ### Set seeds
        seed = self.o_seed + rep_seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        ### For brevity
        k, nP, nQ, N = self.k, self.n['P'], self.n['Q'], self.N
        
        ### --- k=0: logistic regression --- ###
        if self.k == 0:
            start = time.time()
            y = np.concatenate([np.zeros(nP), np.ones(nQ)]).astype(int)

            ### Select the data to use for optimization
            if self.smpl_type == 'original':
                X = self.smpl['All'].detach().clone()
            elif self.smpl_type == 'centered':
                X = self.smpl_cent['All'].detach().clone()
            elif self.smpl_type == 'box_2':
                X = self.smpl_box_2['All'].detach().clone()

            LR_model = linear_model.LogisticRegression(penalty = None, solver = 'newton-cg', multi_class = 'ovr')
            clf      = LR_model.fit(X, y)
            pred     = torch.tensor(clf.predict(X)).to(dtype = torch.float)
            tildeP   = pred[:self.n['P']]
            tildeQ   = pred[self.n['P']:]
            assert len(tildeP) == self.n['P']
            assert len(tildeQ) == self.n['Q']
            ipm = torch.abs(torch.mean(tildeP) - torch.mean(tildeQ))
                    
            return ipm, pred
        ### ----- End of "k=0: logistic regression" --- ###
            
        ### --- k>0: solve the optimizatino problem --- ###
        best_ipm, best_ipm_one = float("-inf"), float("-inf")
        relu = nn.ReLU()
        lin = nn.Linear(self.d, self.N, device = self.device)
        a = torch.normal(0, 1, size = (1, self.N)).squeeze().to(device = self.device)
        
        ### Select the data to use for optimization
        if self.smpl_type == 'original':
            sP, sQ = self.smpl['P'], self.smpl['Q']
        elif self.smpl_type == 'centered':
            sP, sQ = self.smpl_cent['P'], self.smpl_cent['Q']
        elif self.smpl_type == 'box_2':
            sP, sQ = self.smpl_box_2['P'], self.smpl_box_2['Q']   
            
        sP, sQ = sP.to(device = self.device), sQ.to(device = self.device)        

        ### Initialize parameters (w, b, and a)
        with torch.no_grad(): 
            lin.weight.data = F.normalize(lin.weight, p=2, dim=1).to(device = self.device)
            lin.bias.data   = -torch.rand(self.N).to(device = self.device)
            
        a.requires_grad, lin.weight.requires_grad, lin.bias.requires_grad = True, True, True

        ### Initialize optimizer
        if self.optimizer == 'SGD':
            optimizer = optim.SGD([lin.weight, lin.bias, a],
                                   momentum = self.momentum, dampening  = self.momentum,
                                   lr = self.lr)
        elif self.optimizer == 'ADAM':
            optimizer = optim.Adam([lin.weight, lin.bias, a],
                                    betas = (0.9, 0.99), lr = self.lr)
        else:
            print("Unimplemented optimizer.")
            return
        
        ### Print brief information of setting
        txt = f"task: {self.task:<10} / v: {self.v :<2} / N: {self.N} / k : {self.k:<2} / d: {self.d :<3} / nAll: {self.n['All'] :<5} / lr: {self.lr:<5} / steps = {self.tAll:<7}"
        self.PrintIf(txt, end=" ")
        
        ############################################################
        ######## SOLVE OPTIMZIATION PROBLEM FOR k > 0 START ########
        ############################################################
        epsilon = 1e-32 # Due to log, when the value is 0, change it to epsilon
        start = time.time()        
        for i in range(self.tAll):
            optimizer.zero_grad()
            
            ### (1) Calculate MMD or log(MMD) value: without penalty part
            yP = torch.sum(torch.pow(relu(lin(sP)), k), dim = 0) / nP
            yQ = torch.sum(torch.pow(relu(lin(sQ)), k), dim = 0) / (-nQ)
            y  = torch.sum(torch.mul(yP + yQ, a))
            y  = torch.mul(y, 1/N)
            
            if use_log == True:
                y = torch.abs(y)
                if y == 0.0:
                    y  = -torch.log(y + epsilon) / k
                else:
                    y  = -torch.log(y) / k
            else:
                y = -torch.abs(y)/k
            
            ### (2) Calculate penalty part
            w_2_k = torch.pow(torch.sum(torch.pow(lin.weight, 2), 1), k/2)
            # : w_2_k gives gives $||w_i||_2^k$ : N-length row vector
            penalty = torch.mul(torch.sum(torch.mul(torch.abs(a), w_2_k)), 1/N)
            
            if use_log == False:
                # : When log(MMD) is not used, and MMD is used instead,
                #   then use the squared penalty term due to growth rate of MMD. Details in Appendix.
                penalty = torch.pow(penalty, 2) 

            penalty = (self.lamb /  k) * penalty # Adjust the penalty term. Details in Appendix.
            obj = y + penalty # Final objective

            ### (3) Record history
            self.opt_hist.append(obj.item())
            self.opt_nuisance['y'].append(y.item())
            self.opt_nuisance['penalty'].append(penalty.item())
            
            ### (4) Backward pass
            obj.backward()
            optimizer.step()
            
            ### (5) Project bias term to [0, \infty]
            with torch.no_grad():
                lin.bias.data = torch.clamp(lin.bias, max = 0)
                    
            ### (6) Record best 
            with torch.no_grad():
                # (6-1) Normalized version of parameters
                w_temp, b_temp = lin.weight.data.detach().clone(), lin.bias.data.detach().clone()
                a_temp         = a.data.detach().clone()
                
                w_2_1     = torch.pow(torch.sum(torch.pow(w_temp, 2), 1), 1/2)
                w_2_k     = torch.pow(w_2_1, self.k)
                sum_a_w2k = torch.sum(torch.mul(torch.abs(a_temp), w_2_k))

                w_temp, b_temp = F.normalize(w_temp, p=2, dim=1), torch.div(b_temp, w_2_1)
                a_temp         = torch.div(torch.mul(a_temp, w_2_k), sum_a_w2k)

                # (6-2) MMD value with using all neurons
                ipm, _ = self.IpmGivenWB(w_temp, b_temp, a_temp, mult_neurons = True)
                if ipm == 0: ipm = 1e-32
                else: ipm = abs(ipm)
                    
                # (6-3) MMD value with using one best neuron among all neurons
                ipm_one, w_temp_one, b_temp_one = self.OptChooseOne(wb_pool = 'opt_realtime',
                                                                    wb_cand = {'w': w_temp, 'b': b_temp})
                
                # (6-4) Retrieve scaling in certain data type
                if self.smpl_type == 'box_2':
                    ipm = ipm * (self.mean_L2.item() ** self.k) 
                
                # (6-5) Update the best: using all neuron
                if best_ipm <= ipm:
                    best_i = i
                    best_w, best_b, best_a = w_temp, b_temp, a_temp
                    best_ipm = ipm
                    
                # (6-6) Update the best: using one best neuron
                if best_ipm_one <= ipm_one:
                    best_i_one = i
                    best_w_one, best_b_one = w_temp_one, b_temp_one
                    best_ipm_one = ipm_one
                
                # (6-7) Save at checkpoints (if any checkpoint exists)
                if len(checkpoints) > 0:
                    if i+1 in checkpoints:
                        self.opt_checkpoint['best'][i+1]    = {'ipm': best_ipm,
                                                                 'w': best_w,
                                                                 'b': best_b,
                                                                 # 'a': best_a,
                                                                 'i': best_i}
                            
                        self.opt_checkpoint['best_one'][i+1] = {'ipm': best_ipm_one,
                                                                  'w': best_w_one,
                                                                  'b': best_b_one,
                                                                  # 'a': best_a_one,
                                                                  'i': best_i_one}
            ### --- End of "(6) Record best" --- ###
            a.requires_grad, lin.weight.requires_grad, lin.bias.requires_grad = True, True, True
            
            ### (7) Print
            if self.verbose:
                if (i % self.tB) == 0:
                    print(f"\n{'-'*50}")
                    print(f"step {i :< 6} / obj = {round(obj.item(), 4): < 10}")
                    print(f"<w>:\n{lin.weight.data}")
                    print(f"<b>:\n{lin.bias.data}")
                    print(f"<a>:\n{a}")
       
        ##########################################################
        ######## SOLVE OPTIMZIATION PROBLEM FOR k > 0 END ########
        ##########################################################
       
        ### Save results of optimization
        self.opt_time     = round(time.time() - start, 3)
        self.opt_best     = {'ipm': best_ipm, 'w': best_w, 'b': best_b, 'a': best_a, 'i': best_i}
        self.opt_best_one = {'ipm': best_ipm_one, 'w': best_w_one, 'b': best_b_one, 'i': best_i_one}
        self.opt_last     = {'obj': y, 'w': lin.weight.data, 'b': lin.bias.data, 'a': a.data, 'i':self.tAll-1, 'lin': lin}
        
        self.PrintIf(f"/ Time = {self.opt_time}")
    
    #----------------------------------------
    def OptVis(self, plot_type:str = 'history', start = 3):
        """
        Visualize the history of the optimization process. This should be called after "OptSolve" method.
        
        Input:
        - plot_type (str): Specify the information to be plotted.
        - start (int): The plot includes the information from 'start' to very end of the optimization steps.
        """
        
        assert plot_type in ['history', 'y', 'penalty', 'all']
        
        if self.k == 0:
            print("k=0 does not use gradient based optimization.")
            return
        
        ### Basic setting for plot
        plt.figure(figsize = (4,4))
        x = list(range(self.tAll))
        plt.xlabel("steps")
        txt1 = f"task: {self.task} / v: {self.v} / N: {self.N} / k : {self.k} / d: {self.d} / n: {self.n['All']}"
        txt2 = f"lr: {self.lr} / steps = {self.tAll}"
        plt.title(txt1 + "\n" + txt2)
    
        # For plot_type
        if plot_type == 'history':
            visdata = self.opt_hist
            plt.ylabel("values of opj. function")
            
        elif plot_type == 'y':
            visdata = self.opt_nuisance['y']
            plt.ylabel("values without penalty")
            
        elif plot_type == 'penalty':
            visdata = self.opt_nuisance['penalty']
            plt.ylabel("values of penalty")
            
        elif plot_type == 'all':
            plt.plot(x[start:], self.opt_hist[start:], color = 'r', label = 'history')
            plt.plot(x[start:], self.opt_nuisance['y'][start:], color = 'b', label = 'y')
            plt.plot(x[start:], self.opt_nuisance['penalty'][start:], color = 'k', label = 'penalty')
            plt.legend()
            return None
        
        plt.plot(x[start:], visdata[start:], label = plot_type)
        plt.legend()

    #----------------------------------------
    def OptNorm(self, ipm_type = 'best'):
        """
        Normalize the parameters obtained in the optimization solver, then calculate the MMD value using them.
        This should be called after "OptSolve" method.
        
        Input:
        - ipm_type (str): Decide which parameters to be used in this method.
        """
        
        assert ipm_type in ['best', 'last', 'best_checkpoint', 'last_checkpoint']
        
        ### Choose the parameter type
        if ipm_type == 'best':              record = self.opt_best
        elif ipm_type == 'last':            record = self.opt_last
        elif ipm_type == 'best_checkpoint': record = self.opt_best_checkpoint
        elif ipm_type == 'last_checkpoint': record = self.opt_last_checkpoint
            
        w, b, a, i = record['w'], record['b'], record['a'], record['i']

        ### Normalize w, b, and a
        w_2_1     = torch.pow(torch.sum(torch.pow(w, 2), 1), 1/2)
        w_2_k     = torch.pow(w_2_1, self.k)

        sum_a_w2k = torch.sum(torch.mul(torch.abs(a), w_2_k))

        w, b = F.normalize(w, p=2, dim=1), torch.div(b, w_2_1)
        a    = torch.div(torch.mul(a, w_2_k), sum_a_w2k)
            
        ### Calculate MMD value with normalized parameters
        ipm, _ = self.IpmGivenWB(w, b, a, mult_neurons = True)
        if ipm < 0 : ipm, a = -ipm, -a
        if ipm == 0: ipm = 1e-32
            
        ### Retrieve scaling in certain data type
        if self.smpl_type == 'box_2':
            ipm = ipm * (self.mean_L2.item() ** self.k) 

        ### Save results
        self.opt_norm = {'IPM': ipm, 'log_IPM': np.log(ipm), 'w': w, 'b': b, 'a': a, 'i': i}
        
    #----------------------------------------
    def OptMajor(self, verb_grid = False):
        """
        Print the neurons with 'non-negligible' mass among all neurons used in the optimization.
        This should be called after "OptNorm" method.
        
        Input:
        - verb_grid (bool): Whether printing the result of grid search or not.
        """
        
        if self.k==0:
            print("k=0 does not use gradient based optimization.")
            return None
        
        print(f"{'-'*5} Optimization result {'-'*5}")
        eff_prop, eff_N = 0.0, 0
        
        ### Print information of 'non-negligible' neruons only
        for i in range(self.N):
            prop = abs(self.opt_norm['a'][i].item())
            if prop > 1 / min(40, 10 * self.N) : # negligible or not
                print(f"  w = {self.opt_norm['w'][i].tolist()} / b = {self.opt_norm['b'][i].item()} / a = {prop}")
                eff_prop += prop
                eff_N += 1
        print(f"  IPM = {round(self.opt_norm['IPM'], 4)} / eff_N = {eff_N} / eff_prop = {round(eff_prop, 6)}")

        ### Print result from grid search
        if self.grid_best == None:
            return None

        if verb_grid and (self.d==2):
            print(f"{'-'*5} Grid search result {'-'*5}")
            print(f"  w = {self.grid_best['w'].tolist()} / b = {self.grid_best['b']}")
            print(f"  IPM = {self.grid_best['IPM']}")

            # Compare with grid search
            err = 100 * abs(self.opt_norm['IPM'] - self.grid_best['IPM']) / abs(self.grid_best['IPM'])
            print(f"{'-'*5} Error of Optimiztaion {'-'*5}")
            print(f"  {err} %")
   
    #----------------------------------------
    def OptChooseOne(self, wb_pool = 'opt_norm', wb_cand = None):
        """
        Search the best neuron among all neurons.
        This should be called during or after "OptSolve" method.
        
        Input:
        - wb_pool (str): Decide which w and b to use.
        - wb_cand (dict): Includes the w and b value to use. This is None unless "wb_pool == 'opt_realtime'".
        """
        
        assert wb_pool in ['opt_realtime', 'opt_norm', 'opt_best']
        if wb_pool != 'opt_realtime':
            assert wb_cand is None
            
        best_ipm = float('-inf')

        ### Search best neuron among all neurons
        if wb_pool == 'opt_norm':
            for w, b in zip(self.opt_norm['w'], self.opt_norm['b']):
                w = w.unsqueeze(dim = 0)
                ipm, _ = self.IpmGivenWB(w, b, torch.Tensor([1.])) 
                ipm = abs(ipm)
                if best_ipm < ipm:
                    best_ipm = ipm; best_w = w; best_b = b

        elif wb_pool == 'opt_best':
            for w, b in zip(self.opt_best['w'], self.opt_best['b']):
                w = w.unsqueeze(dim = 0)
                ipm, _ = self.IpmGivenWB(w, b, torch.Tensor([1.])) 
                ipm = abs(ipm)
                if best_ipm < ipm:
                    best_ipm = ipm; best_w = w; best_b = b

        elif wb_pool == 'opt_realtime':
            for w, b in zip(wb_cand['w'], wb_cand['b']):
                w = w.unsqueeze(dim = 0)
                ipm, _ = self.IpmGivenWB(w, b, torch.Tensor([1.])) 
                ipm = abs(ipm)
                if best_ipm < ipm:
                    best_ipm = ipm; best_w = w; best_b = b
            
        ### Adjust some corner cases (for visualization and observation done later)
        if best_ipm == 0:
            best_ipm = 1e-32
            
        if best_ipm == float('-inf'):
            best_ipm = 1e-32; best_w = w; best_b = b
            
        ### Retrieve scaling in certain data type
        if self.smpl_type == 'box_2':
            best_ipm = best_ipm * (self.mean_L2.item() ** self.k)
        
        ### Final output: return or save result
        if wb_pool == 'opt_realtime':
            return best_ipm, best_w, best_b
        else:
            self.opt_one = {'IPM': best_ipm, 'log_IPM': np.log(best_ipm), 'w': best_w, 'b': best_b}
            
    #----------------------------------------
    def OracleTestStatistic(self):
        """
        Calculate the LRT (likelihood ratio test) statistics for different self.task.
        """
        # For brevity
        epsilon = 1e-16
        nP, nQ, nAll = self.n['P'], self.n['Q'], self.n['All']
        sP, sQ, sAll = self.smpl['P'], self.smpl['Q'], self.smpl['All']
        if self.task in ['t-one']:
            mP, mQ = torch.zeros(self.d), torch.zeros(self.d)
        else:
            vP, vQ = self.dist['P'].covariance_matrix, self.dist['Q'].covariance_matrix
            mP, mQ = self.dist['P'].loc, self.dist['Q'].loc
            
        cP, cQ = nP/nAll, nQ/nAll # proportion of nP and nQ
        
        ### LRT for each task
        if self.task == 'mean-shift':    
            tempP, tempQ = torch.pow(sP - mP, 2)  , torch.pow(sQ - mQ, 2)
            tempP, tempQ = torch.sum(tempP, dim=1), torch.sum(tempQ, dim=1)
            
            # lrt_alt
            lrt_alt = -0.5 * torch.sum(tempP) + -0.5 * torch.sum(tempQ)
            
            # lrt _null
            tempAll_P = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mP, 2), dim=1))
            tempAll_Q = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mQ, 2), dim=1))
            lrt_null  = torch.sum(torch.log(cP * tempAll_P + cQ * tempAll_Q + epsilon))
            
            # final test statistic
            oracle_test = lrt_alt - lrt_null
            
        elif self.task == 'var-one':
            tempP, tempQ = torch.pow(sP - mP, 2), torch.pow(sQ - mQ, 2)
            v = vQ[0,0]
            tempQ[:, 0] = tempQ[:, 0] / v
            tempP, tempQ = torch.sum(tempP, dim=1), torch.sum(tempQ, dim=1)
            
            # lrt_alt
            lrt_alt = -0.5 * torch.sum(tempP) + -0.5 * torch.sum(tempQ) + -0.5 * torch.log(v) * nQ
            
            # lrt _null
            tempAll_P       = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mP, 2), dim=1))
            tempAll_Q       = torch.pow(sAll - mQ, 2)
            tempAll_Q[:, 0] = tempAll_Q[:, 0] / v
            tempAll_Q       = torch.exp(-0.5 * torch.sum(tempAll_Q, dim=1)) / (v ** 0.5)
            lrt_null        = torch.sum(torch.log(cP * tempAll_P + cQ * tempAll_Q + epsilon))
            
            # final test statistic
            oracle_test = lrt_alt - lrt_null            

        elif self.task == 'var-all':
            tempP, tempQ = torch.pow(sP - mP, 2)  , torch.pow(sQ - mQ, 2)
            v = vQ[0,0]
            tempP, tempQ = torch.sum(tempP, dim=1), torch.sum(tempQ, dim=1) / v

            # lrt_alt
            lrt_alt = -0.5 * torch.sum(tempP) + -0.5 * torch.sum(tempQ) + -0.5 * torch.log(v) * self.d * nQ
            
            # lrt _null
            tempAll_P = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mP, 2), dim=1))
            tempAll_Q = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mQ, 2), dim=1) / v)
            tempAll_Q = tempAll_Q / (v ** (self.d * 0.5))
            lrt_null  = torch.sum(torch.log(cP * tempAll_P + cQ * tempAll_Q + epsilon))
            
            # final test statistic
            oracle_test = lrt_alt - lrt_null            
                
        elif self.task == 'kmmd-iso':
            tempP, tempQ = sP - mP, sQ - mQ
            diag = torch.pow(torch.diag(vQ), -1)
            tempP, tempQ = torch.mul(torch.pow(tempP, 2), diag), torch.mul(torch.pow(tempQ, 2), diag)
            
            # lrt_alt
            lrt_alt = -0.5 * torch.sum(tempP) + -0.5 * torch.sum(tempQ)
            
            # lrt _null
            tempAll_P = torch.exp(-0.5 * torch.sum(torch.mul(torch.pow(sAll - mP, 2), diag), dim=1))
            tempAll_Q = torch.exp(-0.5 * torch.sum(torch.mul(torch.pow(sAll - mQ, 2), diag), dim=1))
            lrt_null  = torch.sum(torch.log(cP * tempAll_P + cQ * tempAll_Q + epsilon))

            # final test statistic
            oracle_test = lrt_alt - lrt_null
            
        elif self.task == 't-one':
            v = self.dist['Q']['t'].df
            tempP, tempQ = sP - mP, sQ - mQ
            tempP   = torch.sum(torch.pow(tempP, 2), dim=1)
            tempQ_t = torch.tensor(t.pdf(tempQ[:, 0], df = v)).to(dtype = torch.float)
            tempQ_n = torch.sum(torch.pow(tempQ[:, 1:], 2), dim=1)
            
            # lrt_alt
            lrt_alt_P = -0.5 * torch.sum(tempP) - 0.5 * nP * torch.log(torch.tensor(torch.pi))
            lrt_alt_Q = -0.5 * torch.sum(tempQ_n) + torch.sum(torch.log(tempQ_t))
            lrt_alt   = lrt_alt_P + lrt_alt_Q
            
            # lrt_null
            tempAll_P   = torch.exp(-0.5 * torch.sum(torch.pow(sAll - mP, 2), dim=1))
            tempAll_Q   = sAll - mQ
            tempAll_Q_t = torch.tensor(t.pdf(tempAll_Q[:, 0], df = v)).to(dtype = torch.float)
            tempAll_Q_n = torch.exp(-0.5 * torch.sum(torch.pow(tempAll_Q[:, 1:], 2), dim=1))
            tempAll_Q   = tempAll_Q_t * tempAll_Q_n
            lrt_null    = torch.sum(torch.log(cP * tempAll_P + cQ * tempAll_Q + epsilon))
            
            # final test statistic
            oracle_test = lrt_alt - lrt_null
            
        else:
            oracle_test = torch.tensor(float("-inf"))
        
        self.oracle_test = oracle_test.item()
    
    #----------------------------------------
    def RivalTestStatistic(self, rival:str = None, **kwargs):
        """
        Calculate other well-known non-parametric two sample test statistics.
        
        Input:
        - rival (str): The type of non-parametric test.
        - **kwargs: Varies by the value of 'rival'.
        """
        
        assert rival in ['kmmd', 'energy']
        sP, sQ, sAll = self.smpl['P'], self.smpl['Q'], self.smpl['All'] # for brevity

        ### Kernel MMD test
        if rival == 'kmmd':
            degree_poly = kwargs['degree_poly'] if 'degree_poly' in kwargs else 2
            gamma_poly  = kwargs['gamma_poly'] if 'gamma_poly' in kwargs else 1.0
            gamma_gaus  = kwargs['gamma_gaus'] if 'gamma_gaus' in kwargs else 1.0
            coef0       = kwargs['coef0'] if 'coef0' in kwargs else 0.0

            kmmd_out  = {'linear': -1, "quadratic": -1, "cubic":-1, "gaussian-est": -1} 
            kmmd_time = {'linear': -1, "quadratic": -1, "cubic":-1, "gaussian-est": -1}

            for key in kmmd_out.keys():
                start_key = time.time()
                
                if key == "quadratic": kernel, degree_poly = "polynomial", int(2)
                elif key == "cubic":   kernel, degree_poly = "polynomial", int(3)
                else:
                    kernel = key
                    
                kmmd_out[key]  = MMD_calculator(sP, sQ, kernel, degree_poly, gamma_poly, gamma_gaus, coef0)
                kmmd_time[key] = round(time.time() - start_key, 6)
            
            return kmmd_out, kmmd_time
        
        ### Energy distance test
        elif rival == 'energy':
            x, y = sP.numpy(), sQ.numpy()
            energy_out = dcor.homogeneity.energy_test_statistic(x, y)
            return energy_out
        
        else:
            return None
    
    #----------------------------------------
    def AltNullRepeat(self, rep_sample:int = None, rep_optim:int = None,
                            sample_key = float("nan"), optim_key = float("nan"),
                            altnull_task = None, check_tIs:list = []):

        """
        For both alternative and null hypothesis, resample the data certain amount of times, and calculate (sample) MMD value per sample.
        
        Input:
        - rep_sample (int): How many times the resampling happens.
        - rep_optim (int): How many times the optimization is solved with different initialized parameters.
        - sample_key (int): Key for the sample. Nuisance information. Can be used in the later data processing.
        - optim_key (int): Key for the optimization. Nuisance information. Can be used in the later data processing.
        - altnull_task (int): Specify whether the optimization uses both log and no-log version, or just uses the log version.
        - check_tIs (list): Specify the checkpoints.
        
        Output:
        - output (pd.DataFrame)
        """
        
        # Basic check and setup
        assert altnull_task in ['lognolog', 'logonly']
        assert rep_sample is not None
        assert rep_optim is not None
        self.rep_sample = rep_sample
        self.rep_optim  = rep_optim
        
        if len(check_tIs) == 0: check_tIs = [self.tI]

        checkpoints = [tI * self.tB for tI in check_tIs]
        
        # Generate distribution: this does not affect by rep_sample or rep_optim
        self.GenDist()
        
        # Set seeds
        random.seed(self.s_seed)
        seed_dict = { 'alt'  : {'sample': random.sample(range(5 * self.rep_sample), self.rep_sample),
                                'optim' : random.sample(range(5 * self.rep_optim) , self.rep_optim ) },
                      'null' : {'sample': random.sample(range(5 * self.rep_sample), self.rep_sample),
                                'optim' : random.sample(range(5 * self.rep_optim) , self.rep_optim ) } }
        
        ### Initialize output dataframe and column names
        output = []
        
        cols = ['hypo', 'rep', 'oracle', 'kmmd_1', 'kmmd_2', 'kmmd_3', 'kmmd_g', 'energy']
        cols = cols +['log_nolog', "eff_tI", "ipm_max", "ipm_max_one"]    
        cols = cols + [f"ipm_optim:{s_rep}" for s_rep in range(rep_optim)]
        cols = cols + [f"ipm_one:{s_rep}"   for s_rep in range(rep_optim)]
        
        ### --- Repetition starts --- ###
        for hypo in ['alt', 'null']:
            
            if altnull_task == 'lognolog': 
                loc_init = int(0) if hypo == 'alt' else int(2*self.rep_sample* len(check_tIs)) # for later ".loc"
            elif altnull_task == 'logonly':
                loc_init = int(0) if hypo == 'alt' else int(self.rep_sample* len(check_tIs))   # for later ".loc"
            
            ### --- (0) Repeat sampling starts --- ###
            for i in range(self.rep_sample):
                rep = i+1                            
                
                ### (1) Get one sample (nP data from distribution P, and nQ data from distribution Q)
                rep_seed_s = seed_dict[hypo]['sample'][i]
                self.GenSmpl(HYPO = hypo, rep_seed = rep_seed_s)
                
                ### (2) Solve optimization problem multiple times 
                ipm_log,   ipm_log_one   = {tI:[] for tI in check_tIs}, {tI:[] for tI in check_tIs}
                ipm_nolog, ipm_nolog_one = {tI:[] for tI in check_tIs}, {tI:[] for tI in check_tIs}
                
                ipm_max_log, ipm_max_log_one, ipm_max_nolog, ipm_max_nolog_one  = {}, {}, {}, {}
                
                ### (2-1) k>0
                if self.k > 0:
                    for j in range(self.rep_optim):
                        rep_seed_o = seed_dict[hypo]['optim'][j]
                        
                        # for log (i.e. log(MMD) + penalty)
                        self.OptSolve(rep_seed = rep_seed_o, use_log = True, checkpoints = checkpoints)
                        for tI in check_tIs:
                            ipm_log[tI].append(self.opt_checkpoint['best'][tI*self.tB]['ipm'])
                            ipm_log_one[tI].append(self.opt_checkpoint['best_one'][tI*self.tB]['ipm'])
                        
                        # for nolog (i.e. MMD + penalty)
                        if altnull_task == 'lognolog':
                            self.OptSolve(rep_seed = rep_seed_o, use_log = False, checkpoints = checkpoints)
                            for tI in check_tIs:
                                ipm_nolog[tI].append(self.opt_checkpoint['best'][tI*self.tB]['ipm'])
                                ipm_nolog_one[tI].append(self.opt_checkpoint['best_one'][tI*self.tB]['ipm'])
                    
                    # Get max MMD value from multiple optimization solutions
                    for tI in check_tIs:
                        ipm_max_log[tI]       = np.max(ipm_log[tI])
                        ipm_max_log_one[tI]   = np.max(ipm_log_one[tI])

                        if altnull_task == 'lognolog':
                            ipm_max_nolog[tI]     = np.max(ipm_nolog[tI])
                            ipm_max_nolog_one[tI] = np.max(ipm_nolog_one[tI])
                
                ### (2-2) k=0: do not solve optimzation problem multiple times, since it uses logistic regression
                else:
                    ipm, _ = self.OptSolve()
                    
                    # Since optimization is not solved multiple times, fill the output dataframe with the same values
                    for tI in check_tIs:
                        ipm_log[tI]       = [float(ipm)] * self.rep_optim
                        ipm_log_one[tI]   = ipm_log[tI]
                        if altnull_task == 'lognolog':
                            ipm_nolog[tI]     = ipm_log[tI]
                            ipm_nolog_one[tI] = ipm_log[tI]
                    
                    for tI in check_tIs:
                        ipm_max_log[tI]       = float(ipm)
                        ipm_max_log_one[tI]   = float(ipm)
                        if altnull_task == 'lognolog':
                            ipm_max_nolog[tI]     = float(ipm)
                            ipm_max_nolog_one[tI] = float(ipm)
                    
                ### (3) Oracle test statistics
                self.OracleTestStatistic()
                oracle_test  = self.oracle_test
                
                ### (4) Kernel MMD test statistics
                kmmd_out, kmmd_time = self.RivalTestStatistic('kmmd')
                kmmd_lin, kmmd_quad, kmmd_cube, kmmd_gaus = kmmd_out['linear'], kmmd_out['quadratic'], kmmd_out['cubic'], kmmd_out['gaussian-est']
                
                ### (5) Energy distance test statistics
                energy_stat  = self.RivalTestStatistic('energy')
                
                ### (6) Save results to output
                for tI in check_tIs:
                    shared = [f"{hypo}_hypo", rep, oracle_test, kmmd_lin, kmmd_quad, kmmd_cube, kmmd_gaus, energy_stat]
                    
                    row_log = shared + ['log', tI, ipm_max_log[tI], ipm_max_log_one[tI]]
                    row_log = row_log + ipm_log[tI] + ipm_log_one[tI]
                    output.append(row_log)
                    
                    if altnull_task == 'lognolog':
                        row_nolog = shared + ['nolog', tI, ipm_max_nolog[tI], ipm_max_nolog_one[tI]]
                        row_nolog = row_nolog + ipm_nolog[tI] + ipm_nolog_one[tI]
                        output.append(row_nolog)
                    
            ### --- (0) Repeat sampling ends --- ###
        ### ---Repetition ends --- ###
        
        output = pd.DataFrame(output, columns=cols)
        return output
    
    #----------------------------------------
    def MMDCurveSolve(self, rep_seed = 0, use_log = True):
        """
        Solve optimization function, and output all the history of the MMD value throughout the whole optimization process.
        This method is similar to "OptSolve" method, but some parts are modified.
        
        Input:
        - rep_seed (int): When the same optimization problem is solved several times, to change the initialization of the parameter.
        - use_log (bool): When it is true, solve 'log(MMD) + penalty'. When it is False, solve 'MMD + penalty'
        """
        
        # Basic initialization
        output, output_one = [], []
        seed = self.o_seed + rep_seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        ### For brevity
        k, nP, nQ, N = self.k, self.n['P'], self.n['Q'], self.N
        
        # Unlike self.OptSolve, MMDCurveSolve only cares k>0 case
        assert k > 0   
        best_opt = float("inf")
        relu = nn.ReLU()
        lin = nn.Linear(self.d, self.N, device = self.device)
        a = torch.normal(0, 1, size = (1, self.N)).squeeze().to(device = self.device)
        
        ### Select the data to use for optimization
        if self.smpl_type == 'original':
            sP, sQ = self.smpl['P'], self.smpl['Q']
        elif self.smpl_type == 'centered':
            sP, sQ = self.smpl_cent['P'], self.smpl_cent['Q']
        elif self.smpl_type == 'box_2':
            sP, sQ = self.smpl_box_2['P'], self.smpl_box_2['Q']   
            
        sP, sQ = sP.to(device = self.device), sQ.to(device = self.device)    

        ### Initialize parameters (w, b, and a)
        with torch.no_grad(): 
            lin.weight.data = F.normalize(lin.weight, p=2, dim=1).to(device = self.device)
            lin.bias.data   = -torch.rand(self.N).to(device = self.device)
            
        a.requires_grad, lin.weight.requires_grad, lin.bias.requires_grad = True, True, True

        ### Initialize optimizer
        if self.optimizer == 'SGD':
            optimizer = optim.SGD([lin.weight, lin.bias, a],
                                   momentum = self.momentum, dampening  = self.momentum,
                                   lr = self.lr)
        elif self.optimizer == 'ADAM':
            optimizer = optim.Adam([lin.weight, lin.bias, a],
                                    betas = (0.9, 0.99), lr = self.lr)
        else:
            print("Unimplemented optimizer.")
            return
        
        ### Print brief information of setting
        txt = f"task: {self.task:<10} / v: {self.v :<2} / N: {self.N} / k : {self.k:<2} / d: {self.d :<3} / nAll: {self.n['All'] :<5} / lr: {self.lr:<5} / steps = {self.tAll:<7}"
        self.PrintIf(txt, end=" ")
        
        ############################################################
        ######## SOLVE OPTIMZIATION PROBLEM FOR k > 0 START ########
        ############################################################
        epsilon = 1e-32 # Due to log, when the value is 0, change it to epsilon
        start = time.time()         
        for i in range(self.tAll):
            optimizer.zero_grad()
            
            ### (1) Calculate MMD or log(MMD) value: without penalty part
            yP = torch.sum(torch.pow(relu(lin(sP)), k), dim = 0) / nP
            yQ = torch.sum(torch.pow(relu(lin(sQ)), k), dim = 0) / (-nQ)
            y  = torch.sum(torch.mul(yP + yQ, a))
            y  = torch.mul(y, 1/N)
            
            if use_log == True:
                y = torch.abs(y)
                if y == 0.0:
                    y  = -torch.log(y + epsilon) / k
                else:
                    y  = -torch.log(y) / k
            else:
                y = -torch.abs(y)/k
            
            ### (2) Calculate penalty part
            w_2_k = torch.pow(torch.sum(torch.pow(lin.weight, 2), 1), k/2)
            # : w_2_k gives gives $||w_i||_2^k$ : N-length row vector
            penalty = torch.mul(torch.sum(torch.mul(torch.abs(a), w_2_k)), 1/N)
            
            if use_log == False:
                # : When log(MMD) is not used, and MMD is used instead,
                #   then use the squared penalty term due to growth rate of MMD. Details in Appendix.
                penalty = torch.pow(penalty, 2) 

            penalty = (self.lamb /  k) * penalty # Adjust the penalty term. Details in Appendix.
            obj = y + penalty # Final objective

            ### (3) Record history: this is not required in MMDCurveSolve
            
            #### (4) Backward pass
            obj.backward()
            optimizer.step()
            
            ### (5) Project bias term to [0, \infty]
            with torch.no_grad():
                lin.bias.data = torch.clamp(lin.bias, max = 0)
                
            ### (6) Record best: this is not required in MMDCurveSolve.
            ### (7) Instead, save the MMD value throughout whole optimization process to output and output_one
            with torch.no_grad():
                # (7-1) Normalized version of parameters
                w_temp, b_temp = lin.weight.data.detach().clone(), lin.bias.data.detach().clone()
                a_temp         = a.data.detach().clone()
                
                w_2_1     = torch.pow(torch.sum(torch.pow(w_temp, 2), 1), 1/2)
                w_2_k     = torch.pow(w_2_1, self.k)
                sum_a_w2k = torch.sum(torch.mul(torch.abs(a_temp), w_2_k))

                w_temp, b_temp = F.normalize(w_temp, p=2, dim=1), torch.div(b_temp, w_2_1)
                a_temp         = torch.div(torch.mul(a_temp, w_2_k), sum_a_w2k)

                # (7-2) MMD value with using all neurons
                ipm, _ = self.IpmGivenWB(w_temp, b_temp, a_temp, mult_neurons = True)
                if ipm == 0: ipm = 1e-32
                else: ipm = abs(ipm)

                # (7-3) MMD value with using one best neuron among all neurons
                ipm_one, w_temp_one, b_temp_one = self.OptChooseOne(wb_pool = 'opt_realtime',
                                                                    wb_cand = {'w': w_temp, 'b': b_temp})
                
                # (7-4) Retrieve scaling in certain data type (cf. 7-3 already considered the scaling)
                if self.smpl_type == 'box_2':
                    ipm = ipm * (self.mean_L2.item() ** self.k) 
                
                # (7-5) Save results to output and output_one
                output.append(ipm)
                output_one.append(ipm_one)
            # ----- End of (7) ----- #
        
            a.requires_grad, lin.weight.requires_grad, lin.bias.requires_grad = True, True, True
        ##########################################################
        ######## SOLVE OPTIMZIATION PROBLEM FOR k > 0 END ########
        ##########################################################
        
        return output, output_one
    
    #----------------------------------------
    def MMDCurveAltNullRepeat(self, rep_sample:int = None, rep_optim:int = None,
                                    sample_key = float("nan"), optim_key = float("nan"),
                                    check_tIs = []):
        """
        For both alternative and null hypothesis, resample the data certain amount of times, and save all MMD values calculated throughout the optimization process.
        This method is similar to "AltNullRepeat" method:
            While "AltNullRepeat" use "OptSolve", this method use "MMDCurveSolve" instead.
            Also, this method runs both log (i.e. log(MMD) + penalty) and nolog (i.e. MMD + penalty) optimization.
        
        Input:
        - rep_sample (int): How many times the resampling happens.
        - rep_optim (int): How many times the optimization is solved with different initialized parameters.
        - sample_key (int): Key for the sample. Nuisance information. Can be used in the later data processing.
        - optim_key (int): Key for the optimization. Nuisance information. Can be used in the later data processing.
        
        Output:
        - output (pd.DataFrame)
        """
        
        # Basic check and setup
        assert rep_sample is not None
        assert rep_optim is not None
        self.rep_sample = rep_sample
        self.rep_optim  = rep_optim
        
        if len(check_tIs) == 0: check_tIs = [self.tI]

        checkpoints = [tI * self.tB for tI in check_tIs]
        
        # Generate distribution: this does not affect by rep_sample or rep_optim
        self.GenDist()
        
        # Set seeds
        random.seed(self.s_seed)
        seed_dict = { 'alt'  : {'sample': random.sample(range(5 * self.rep_sample), self.rep_sample),
                                'optim' : random.sample(range(5 * self.rep_optim) , self.rep_optim ) },
                      'null' : {'sample': random.sample(range(5 * self.rep_sample), self.rep_sample),
                                'optim' : random.sample(range(5 * self.rep_optim) , self.rep_optim ) } }
        
        ### Initialize output dataframe and column names
        output = []
        
        cols = ['hypo', 'rep_no:smpl', 'rep_no:optim', 'log_nolog', 'all_or_one', 'gridsearch', 'mmdlast']
        cols = cols + [f'mmdbest:{cp}' for cp in checkpoints]
        cols = cols + [f"mmdcurve:{step}" for step in range(self.tAll)]
        
        ### --- Repetition starts --- ###
        for hypo in ['alt', 'null']: 
            loc_init = int(0) if hypo == 'alt' else int(4 * self.rep_sample * self.rep_optim) # for later ".loc"
            loc_count = 0
            
            ### --- (0) Repeat sampling starts --- ###
            for i in range(self.rep_sample): # repeat sampling
                rep = i+1                            
                
                ### (1) Get one sample (nP data from distribution P, and nQ data from distribution Q)
                rep_seed_s = seed_dict[hypo]['sample'][i]
                self.GenSmpl(HYPO = hypo, rep_seed = rep_seed_s)
                
                ### (2) Grid search if dimension is 2
                if self.d == 2:
                    gdict = {'w_cut': 200 + 1, 'b_bound': 5, 'b_cut': 100 + 1}
                    self.GridSearch(gdict)
                    grid_ipm = self.grid_best['IPM']
                    
                else: grid_ipm = -1.0
                
                ### (3) Solve optimization problem multiple times. Here, only care k>0 case.
                assert self.k > 0
                
                for j in range(self.rep_optim):
                    rep_seed_o = seed_dict[hypo]['optim'][j]
                    
                    # for log (i.e. log(MMD) + penalty)
                    log_mmd,   log_mmd_one    = self.MMDCurveSolve(rep_seed = rep_seed_o, use_log = True)
                    
                    # for nolog (i.e. MMD + penalty)
                    nolog_mmd, nolog_mmd_one  = self.MMDCurveSolve(rep_seed = rep_seed_o, use_log = False)
                    
                    # make rows for dataframe
                    shared = [f"{hypo}_hypo", rep, j+1]
                    
                    row_log = shared + ['log', 'all', grid_ipm, log_mmd[-1]]
                    row_log = row_log + [max(log_mmd[:cp]) for cp in checkpoints] + log_mmd
                    
                    row_log_one = shared + ['log', 'one', grid_ipm, log_mmd_one[-1]]
                    row_log_one = row_log_one + [max(log_mmd_one[:cp]) for cp in checkpoints] + log_mmd_one
                    
                    row_nolog = shared + ['nolog', 'all', grid_ipm, nolog_mmd[-1]]
                    row_nolog = row_nolog + [max(nolog_mmd[:cp]) for cp in checkpoints] + nolog_mmd
                    
                    row_nolog_one = shared + ['nolog', 'one', grid_ipm, nolog_mmd_one[-1]]
                    row_nolog_one = row_nolog_one + [max(nolog_mmd_one[:cp]) for cp in checkpoints] + nolog_mmd_one
                    
                    # save results (which are 4 rows) to output 
                    output.append(row_log)
                    output.append(row_log_one)
                    output.append(row_nolog)
                    output.append(row_nolog_one)
                
        output = pd.DataFrame(output, columns=cols)
        return output
    