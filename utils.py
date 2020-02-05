import torch
import torch.nn as nn
import torchaudio
from scipy.stats import kurtosis

def load(filename):
    """ Load a .wav file into a tensor. """
    audio, sample_rate = torchaudio.load(filename)
    return audio, sample_rate

def save(filename, audio, sample_rate):
    """ Save tensor as audio wav file. """
    torchaudio.save(filename, audio, sample_rate)
    return None

def kurto(y):
    """ Compute kurtosis of signal"""
    numpy_kurto = kurtosis(torch.t(y).detach().numpy())
    return torch.from_numpy(numpy_kurto)

def edgeworth(s, k):
    phi = s - k * (s**3 - 3 * s) / 6.0
    return phi

class H(nn.Module):
    """ Estimation function.
    Attributes:
        - phi : non-linear function.
        - n (int): number of sources.
    """
    def __init__(self, phi, n):
        super(H, self).__init__()
        self.phi = phi
        self.n = n
    
    def forward(self, y_t, k):
        """
        Args:
            - y_t (n,): vector of estimated sources, at time t.
            - k (n,): vector of associated kurtosis.
        """
        H = torch.dot(self.phi(y_t, k), torch.t(y_t)) - torch.eye(self.n)
        return H

class estimation_equation(nn.Module):
    """ Custom loss.
    Args:
        - n: number of sources/mixtures
        - estim_f : estimation function
    """
    def __init__(self, n, estim_f):
        super(estimation_equation, self).__init__()
        self.n = n
        self.H = estim_f
        
    def forward(self, y):
        T = y.size()[1]
        k = kurto(y)
        
        H_hat = torch.zeros((self.n, self.n))
        for t in range(T):
            H_hat += self.H(y[:,t], k)
        
        H_hat /= T
        
        return torch.sum(H_hat)