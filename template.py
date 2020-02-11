import numpy as np
#import XMLReader
import patterns

def voice(j, N, nbin, pan=0):
    """
    Voice model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    
    source = {}
    K = 20 # number of activation patterns
    L = 50 # pitch range
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + 'EstimatedVoice' 
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))
    
    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, L)) + 0.25 * np.ones((nbin, L))
    
    source['Uex'] = {}
    source['Uex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Uex']['data'] = 0.75 * abs(np.random.randn(L, K)) + 0.25 * np.ones((L, K))
    
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'fixed'
    source['Gex']['data'] = np.eye((K, M)) # GSMM model for monophonic
    
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(M, N)) + 0.25 * np.ones((M, N))
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source
    

def guitar(j, N, nbin, pan=0):
    """
    Guitar model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    source = {}
    L = 6 * 60
    K = 20
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + '_EstimatedGuitar'
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))

    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, K)) + 0.25 * np.ones((nbin, K))
    source['Uex'] = {}
    source['Uex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Uex']['data'] = 0.75 * abs(np.random.randn(L, K)) + 0.25 * np.ones((L, K))
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Gex']['data'] = 0.75 * abs(np.random.randn(K, M)) + 0.25 * np.ones((K, M))
    source['Hex'] = {}
    source['Hex']['data'] = patterns.H_attack_decay(M, N)
    source['Hex']['adaptability'] = 'fixed' # Will be adapted by FASST
    
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source

def piano(j, N, nbin, pan=0):
    """
    Piano model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    
    source = {}
    K = 20 # number of activation patterns
    L = 88 # pitch range
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + 'EstimatedPiano' 
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))
    
    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, L)) + 0.25 * np.ones((nbin, L))
    
    source['Uex'] = {}
    source['Uex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Uex']['data'] = 0.75 * abs(np.random.randn(L, K)) + 0.25 * np.ones((L, K))
    
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'free'
    source['Gex']['data'] = 0.75 * abs(np.random.randn(K, M)) + 0.25 * np.ones((K, M))
    
    source['Hex'] = {}
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Hex']['data'] = 0.75 * abs(np.random.randn(M, N)) + 0.25 * np.ones((M, N))
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source

def trumpet(j, N, nbin, pan=0):
    """
    Trumpet model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    
    source = {}
    K = 20 # number of activation patterns
    L = 50 # pitch range
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + 'EstimatedVoice' 
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))
    
    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, L)) + 0.25 * np.ones((nbin, L))
    
    source['Uex'] = {}
    source['Uex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Uex']['data'] = 0.75 * abs(np.random.randn(L, K)) + 0.25 * np.ones((L, K))
    
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'fixed'
    source['Gex']['data'] = np.eye((K, M)) # GSMM model for monophonic
    
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(M, N)) + 0.25 * np.ones((M, N))
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source

def saxophone(j, N, nbin, pan=0):
    """
    Saxophone model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    
    source = {}
    K = 20 # number of activation patterns
    L = 50 # pitch range
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + 'EstimatedSaxophone' 
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))
    
    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, L)) + 0.25 * np.ones((nbin, L))
    
    source['Uex'] = {}
    source['Uex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Uex']['data'] = 0.75 * abs(np.random.randn(L, K)) + 0.25 * np.ones((L, K))
    
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'fixed'
    source['Gex']['data'] = np.eye((K, M)) # GSMM model for monophonic
    
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(M, N)) + 0.25 * np.ones((M, N))
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source

def drums(j, N, nbin, pan=0):
    """
    Drums model template.
    Args:
        - j (int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    
    source = {}
    K = 20 # number of activation patterns
    M = 2 * N
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + 'EstimatedDrums' 
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))
    
    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, K)) + 0.25 * np.ones((nbin, K))
    
    source['Gex'] = {}
    source['Gex']['adaptability'] = 'free'
    source['Gex']['data'] = 0.75 * abs(np.random.randn(K, M)) + 0.25 * np.ones((K, M))
    
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(M, N)) + 0.25 * np.ones((M, N))
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source
    
def other(j, N, nbin, pan=0):
    """NMF-like default template.
    Args:
        - j(int <= J): # of the source.
        - N (int): number of frames.
        - nbin (int): number of frequency bins.
        - pan (float): stereo pan. -1 for left and +1 for right.
    """
    source = {}
    K = 4           # NFM rank (number of spectral patterns in the dictionary)
        
    # Name of output audio file for source j
    source['name'] = {}
    source['name'] = str(j) + '_EstimatedSource'
    
    # Spatial model
    source['A'] = {}
    source['A']['mixingType'] = 'inst'    # Instantaneous mixture
    source['A']['adaptability'] = 'free'  # Will be adapted by FASST
    source['A']['data'] = np.array(([np.sin((1+pan) * np.pi/4)],
                                    [np.cos((1+pan) * np.pi/4)]))

    # Spectral patterns (Wex) and time activation patterns applied to spectral patterns (Hex)
    source['Wex'] = {}
    source['Wex']['adaptability'] = 'free' # Will be adapted by FASST
    source['Wex']['data'] = 0.75 * abs(np.random.randn(nbin, K)) + 0.25 * np.ones((nbin, K))
    
    source['Hex'] = {}
    source['Hex']['data'] = 0.75 * abs(np.random.randn(K, N)) + 0.25 * np.ones((K, N))
    source['Hex']['adaptability'] = 'free' # Will be adapted by FASST
    
    ## Wiener filter parameters
    source['wiener'] = {}
    source['wiener']['a']  = 0             # a  : Over-substraction parameter (in dB) - Default value = 0
    source['wiener']['b']  = 0             # b  : Phase ponderation parameter (between [0,1]) - Default value = 0
    source['wiener']['c1'] = 0             # c1 : Half-time width of the covariance smoothing window ( c1 >= 0) - Default value = 0
    source['wiener']['c2'] = 0             # c2 : Half-frequency width of the covariance smoothing window ( c2 >= 0) - Default value = 0
    source['wiener']['d']  = float("-inf") # d  : Thresholding parameter ( in dB, d <= 0) - Default value = -Inf
    
    return source