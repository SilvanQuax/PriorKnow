import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
class Classifier(nn.Module):
    def __init__(self, predictor):
        super(Classifier, self).__init__()
        self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        correct = (torch.max(F.softmax(y),dim=1)[1] == t).data.numpy()
        accuracy = np.mean(correct)
        loss = F.cross_entropy(y,t)

        # loss = F.mse_loss(y,t)

        # y = F.sigmoid(y)
        # loss = F.binary_cross_entropy(y,t)
        # accuracy = 0
        return loss, accuracy

class Regressor(nn.Module):
    def __init__(self, predictor):
        super(Regressor, self).__init__()
        self.predictor = predictor

    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.mse_loss(y,t)
        accuracy = 0
        return loss, accuracy

# Network definition
class MLP(nn.Module):

    def __init__(self, n_inputs, n_units, n_out, bias = False, idx = None, sig_noise=0):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_units)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_out, bias=bias)  # n_units -> n_out
        self.p1 = nn.Parameter(torch.ones(n_units))  # n_units -> n_out

        self.idx = idx
        self.sig_noise = sig_noise

    def __call__(self, x):
        self.h = F.relu(self.p1*self.l1(x))# + self.sig_noise*np.random.randn(self.l1(x).shape[0],self.l1(x).shape[1])
        if self.idx is not None:
            self.h.data[:,self.idx]=0
        return self.l2(self.h)

# Network definition
class MLP2(nn.Module):

    def __init__(self, n_inputs, n_units, n_out, bias = False, idx = None, sig_noise=0):
        super(MLP2, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_units)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_out, bias=bias)  # n_units -> n_out

        self.idx = idx
        self.sig_noise = sig_noise

    def __call__(self, x):
        self.h = F.relu(self.l1(x))# + self.sig_noise*np.random.randn(self.l1(x).shape[0],self.l1(x).shape[1])
        if self.idx is not None:
            self.h.data[:,self.idx]=0
        return self.l2(self.h)

# Network definition
class MLPCUE(nn.Module):

    def __init__(self, n_inputs, n_units, n_out, bias = False, idx = None, sig_noise=0):
        super(MLPCUE, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_units, bias = True)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_out, bias=bias)  # n_units -> n_out
        self.p1 = nn.Parameter(torch.ones(n_units))  # n_units -> n_out
        self.cue1 = nn.Linear(2, n_units, bias=False)  # n_units -> n_out
        self.cueg1 = nn.Linear(2, n_units, bias=False)  # n_units -> n_out
        nn.init.uniform(self.cue1.weight,a=-1.0*np.sqrt(1.0/52),b=np.sqrt(1.0/52))
        nn.init.uniform(self.cueg1.weight,a=-1.0*np.sqrt(1.0/52),b=np.sqrt(1.0/52))
        # nn.init.uniform(self.cueg1.bias,a=-1.0*np.sqrt(1.0/52),b=np.sqrt(1.0/52))
        # self.cue2 = nn.Linear(2, n_out, bias=bias)  # n_units -> n_out

        self.idx = idx
        self.sig_noise = sig_noise

    def __call__(self, x):
        self.h = F.relu((1+self.cueg1(x[:,-2:]))*(self.l1(x[:,:-2])+self.cue1(x[:,-2:])))# + self.sig_noise*np.random.randn(self.l1(x).shape[0],self.l1(x).shape[1])
        if self.idx is not None:
            self.h.data[:,self.idx]=0
        return self.l2(self.h)#+(1-(x[:,-2:]+0.5)/2)

# Network definition
class MLP_nohid(nn.Module):

    def __init__(self, n_inputs, n_units, n_out, bias = False, idx = None, sig_noise=0):
        super(MLP_nohid, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_out, bias=bias)

        self.idx = idx
        self.sig_noise = sig_noise

    def __call__(self, x):
        if self.idx is not None:
            self.h.data[:,self.idx]=0
        self.h = self.l1(x)
        return self.h

# Network definition
class MLP_nobias(nn.Module):

    def __init__(self, n_inputs, n_units, n_out, bias = False, idx = None, sig_noise=0):
        super(MLP_nobias, self).__init__()
        self.l1 = nn.Linear(n_inputs, n_units, bias=None)  # n_in -> n_units
        self.l2 = nn.Linear(n_units, n_out, bias=bias)  # n_units -> n_out

        self.idx = idx
        self.sig_noise = sig_noise

    def __call__(self, x):
        self.h = F.relu(self.l1(x))# + self.sig_noise*np.random.randn(self.l1(x).shape[0],self.l1(x).shape[1])
        if self.idx is not None:
            self.h.data[:,self.idx]=0
        return self.l2(self.h)