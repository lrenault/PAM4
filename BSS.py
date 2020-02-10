import numpy as np
import matplotlib.pyplot as plt
import wave
import scipy.stats

import librosa
from scipy.io import wavfile
from scipy.stats import kurtosis

filename = 'audios/prise4.wav'
sr_hz, x = librosa.load(filename)

T = sr_hz * 4
n = 2                                   # nbr of sources
x = x[:T]
plt.plot(x)
plt.show()

def writeAudio(name, y, sr_hz):
    normalized_data = np.int16(y/np.max(np.abs(y)) * 32767)
    wavfile.write(name, sr_hz, normalized_data)
    return normalized_data

def whiten(X, method='zca'):
    """
    from https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)

def repatialize(x, angle):
    A = np.array([angle, 1 - angle, angle, 1 - angle]).reshape((2,2))
    s_hat = (A@x.T).T
    writeAudio("audios/mix2.wav", s_hat, sr_hz)
    return s_hat

#%% Cardoso ver.
def phi(s, k):
    """Edgeworth approximation"""
    return s - k * (np.power(s,3) - 3 * s) / 6

def phi_o(s, k):
    return -k * np.power(s,3)

def H_phi(y, k):
    """Estimation function, without whitening"""
    H = np.dot(phi(y, k), y.transpose()) - np.identity(n)
    return H

def H_phi_o(y, k):
    H_o = np.dot(y, y.T) - np.identity(n) + np.dot(phi_o(y,k), y.T) - np.dot(y, phi_o(y,k).T)
    return H_o

def estimation_eq(y, k):
    T = y.shape[0]
    H_hat = np.zeros((n,n))
    for t in range(T):
        H_hat += H_phi(y[t], k)
    return H_hat / T

def relative_gradient_descend(x, epsilon=1e-3, learning_rate=5e-3):
    """Off-line implementation"""
    # whitening
    y = whiten(x)
    
    # first compute
    k = kurtosis(y)
    H_hat = estimation_eq(y,k)
    
    # iterations
    kompteur = 0
    loss = []
    while H_hat.all() > epsilon * np.ones((n,n)).all():
        H_hat = estimation_eq(y,k)

        y -= learning_rate * (H_hat@y.T).T
        k = kurtosis(y)
        
        # log and break
        print(H_hat)
        kompteur += 1
        loss.append(np.sum(H_hat))
        if (kompteur > 3 and loss[-1] > loss[-2]) or kompteur > 200:
            break
        
    plt.plot(loss)
    plt.show()
    return y
#%%
s_hat = relative_gradient_descend(x)

writeAudio("audios/originalL.wav", x[:,0], sr_hz)
writeAudio("audios/originalR.wav", x[:,1], sr_hz)
writeAudio("audios/source1.wav", s_hat[:,0], sr_hz)
writeAudio("audios/source2.wav", s_hat[:,1], sr_hz)
writeAudio("audios/mono.wav", np.sum(s_hat, axis=1), sr_hz)
