import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.stats import kurtosis

filename = 'audios/mix.wav'
sr_hz, x = wavfile.read(filename)

T = sr_hz * 4
n = 2                                   # nbr of sources
x = x[:T].transpose()


def writeAudio(name, y, sr_hz):
    normalized_data = np.int16(y/np.max(np.abs(y)) * 32767)
    wavfile.write(name, sr_hz, normalized_data)
    return normalized_data

#%% algorithm

# 1. Covariance matrix R_xx
#R_xx = np.cov(x)
# 2. Diagonalisation R_xx = Q Sigma Q^T, with Sigma = diag(lmabda,)
# 3. S = Q[:,:K] Sigma[:K, :K]
# 4. Blanchisseur W = hermitian(S)
# 5. z = W x
# 6. estimation U = min contrast^o
# 7. y = U.transpose() $ z
#%% Cardoso ver.
def phi(s, k):
    """Edgeworth approximation"""
    return s - k * (np.power(s,3) - 3 * s) / 6

def H_phi(y, k):
    """Estimation function, without whitening"""
    H = np.dot(phi(y, k), y.transpose()) - np.identity(n)
    return H

def estimation_eq(y, k):
    T = np.shape(y)[1]
    H_hat = np.zeros((n,n))
    for t in range(T):
        H_hat += H_phi(y[:,t], k)
    return H_hat / T

def relative_gradient_descend(x, epsilon=1e-3, learning_rate=1e-3):
    """Off-line implementation"""
    # init
    y = np.copy(x)               # source estimation
    
    # first compute
    k = kurtosis(y.transpose())
    H_hat = estimation_eq(y,k)
    
    # iterations
    kompteur = 0
    while H_hat.all() > epsilon * np.ones((n,n)).all():
        H_hat = estimation_eq(y,k)

        #y -= learning_rate * np.dot(H_hat, y)
        np.add(y, -learning_rate * np.dot(H_hat, y), out=y, casting="unsafe") # a+=b
        k = kurtosis(y.transpose())
        
        # log and break
        print(np.sum(H_hat), k)
        kompteur += 1
        if kompteur > 50:
            return y

    return y
#%%
s_hat = relative_gradient_descend(x)
#%%
writeAudio("audios/source1.wav", s_hat[0], sr_hz)
writeAudio("audios/source2.wav", s_hat[1], sr_hz)
writeAudio("audios/mono.wav", np.sum(s_hat, axis=0), sr_hz)
