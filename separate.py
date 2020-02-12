#!/usr/bin/env python3
#
# Parameters used to initialize FASST in this example:
# * Mixture type : instantaneous.
# * Time-Frequency representation : STFT with 1024 frequency bins.
# * Source paramater Wex : Normally distributed random matrix (default init).
# * Source paramater Hex : Normally distributed random matrix (default init).
# * Source paramater A : balanced gains (left, middle, right)
# * Source paramater adaptability : free, all previous parameters are updated during the iterative EM process.
# * Number of EM iterations : 200
#
###########################################################################
# Copyright 2018 Ewen Camberlein (INRIA), Romain Lebarbenchon (INRIA)
# This software is distributed under the terms of the GNU Public License
# version 3 (http://www.gnu.org/licenses/gpl.txt)
###########################################################################

from __future__ import division
import numpy as np
import os, sys
import wave
import template
#import shutil
import argparse

def main(filename, J, Niteration_EM,
         nb_voice, nb_guitar, nb_piano, nb_trump, nb_sax, nb_drum,
         pan1, pan2, pan3, pan4, pan5, pan6, pan7, pan8):
    """
    Main function for source separation estimation.
    Args:
        - filename (str): path to the mixtures .wav file.
        - J (int): number of sources to estimate.
        - Niteration_EM (int): number of iteration of EM algorithm.
        - nb_voice (int): number of voice tracks in the mixtures.
        - nb_guitar (int): number of guitare tracks in the mixtures.
        - nb_piano (int): number of piano tracks in the mixtures.
        - nb_trump (int): number of trumpet tracks in the mixtures.
        - nb_sax (int): number of saxophone tracks in the mixtures.
        - nb_drum (int): number of drums tracks in the mixtures.
        - pan1-8 (float): panoramic of the j-th source in thr mixtures.

    """
    # ------------------------------------------------------------------------
    #                      Paths management
    # ------------------------------------------------------------------------

    # Path of current file
    script_path = os.path.dirname(os.path.abspath(__file__))
    print(script_path)
    
    # Import the fasst package
    fasst_python_dir = '/usr/local/FASST_2.2.2/scripts/python'
    if fasst_python_dir not in sys.path:
        sys.path.insert(0, fasst_python_dir)
    import fasst
    
    # Create temp/ and result/ directory if it does not exist
    tmp_dir = os.path.join(script_path,'temp/');
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    results_dir = os.path.join(script_path, 'results/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # ------------------------------------------------------------------------
    #                   Mixture and audio scene information
    # ------------------------------------------------------------------------
    mixture_wavname = os.path.join(script_path, filename)
    fid = wave.open(mixture_wavname)
    
    
    # Number of channels
    I = fid.getnchannels()
    
    # Number of mixture samples
    nbSamples_Mix = fid.getnframes()
    
    # ------------------------------------------------------------------------
    #               FASST Initialization (compute input xml file)
    # ------------------------------------------------------------------------
    
    print ('> FASST initialization')
    
    # --- FASST general configuration (direct inputs for FASST)
    
    transformType     = 'STFT'  # Time-frequency transform
    wlen              = 1024    # Window length in samples (frame length in time domain) - should be multiple of 4 for STFT    
    
    # --- Initialization of models and FASST specific parameters for each source
    
    # Define necessary parameters
    N    = int(np.ceil(nbSamples_Mix/wlen*2))  # Number of frames
    nbin = int(wlen/2 + 1)                     # Number of frequency bins for STFT
    pans = np.array([pan1, pan2, pan3, pan4, pan5, pan6, pan7, pan8])
    pans = np.pad(pans, (0, max(0, J - len(pans))))
    
    sources = []
    j = 0
    
    for i in range(nb_voice):
        sources.append(template.voice(j, N, nbin, pans[j]))
        j += 1
    for i in range(nb_guitar):
        sources.append(template.guitar(j, N, nbin, pans[j]))
        j += 1
    for i in range(nb_piano):
        sources.append(template.piano(j, N, nbin, pans[j]))
        j += 1
    for i in range(nb_trump):
        sources.append(template.trumpet(j, N, nbin, pans[j]))
        j += 1
    for i in range(nb_sax):
        sources.append(template.saxophone(j, N, nbin, pans[j]))
        j += 1
    for i in range(nb_drum):
        sources.append(template.drums(j, N, nbin, pans[j]))
        j += 1
    for i in range(J - j):
        sources.append(template.other(j + i, N, nbin, pans[j + i]))    
    
    # Define FASST_data structure
    FASST_data = {}
    FASST_data['tfr_type']   = transformType
    FASST_data['wlen']       = wlen
    FASST_data['iterations'] = Niteration_EM
    FASST_data['sources']    = sources
    
    # Write to XML
    xml_fname = os.path.join(tmp_dir,'sources.xml');
    fasst.writeXML(xml_fname, FASST_data)
    
    # ------------------------------------------------------------------------
    #                        Call FASST binaries
    # ------------------------------------------------------------------------
    print ('> FASST execution')
    
    print ('>> Input time-frequency representation')
    fasst.compute_mixture_covariance_matrix(mixture_wavname, xml_fname, tmp_dir)
    
    print ('>> Refinement of sources models (EM algorithm)')
    fasst.estimate_source_parameters(xml_fname, tmp_dir, xml_fname + '.new')
    
    print ('>> Computation of estimated sources')
    fasst.estimate_sources(mixture_wavname, xml_fname + '.new',tmp_dir, results_dir)
    
    return 0

#%% main call
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Informed audio source separation.')

    parser.add_argument('filename', 
                        type=str, help='Mixtures wave file.')
    
    parser.add_argument('-J', '--nb_sources', default=2,
                        type=int, help='Number of sources.', dest='j')
    
    parser.add_argument('-epoch', '--nb_epoch', default=200,
                        type=str, help="Number of iteration of EM algorithm.", dest='epoch')
    
    parser.add_argument('-v', '--voice', default=0,
                        type=int, help="Number of voice tracks in the mixtures.", dest='v')
    
    parser.add_argument('-g', '--guitar', default=0,
                        type=int, help='Number of guitar tracks in the mixtures.', dest='g')
    
    parser.add_argument('-p', '--piano', default=0,
                        type=int, help="Number of piano tracks in the mixtures.", dest='p')
    
    parser.add_argument('-t', '--trumpet', default=0,
                        type=int, help="Number of trumpet tracks in the mixtures.", dest='t')
    
    parser.add_argument('-s', '--saxophone', default=0,
                        type=int, help="Number of saxophone tracks in the mixtures.", dest='s')
    
    parser.add_argument('-d', '--drums', default=0,
                        type=int, help="Number of drum tracks in the mixtures.", dest='d')
    
    parser.add_argument('-pan1', default=0, type=float, help="Pan of 1st source.", dest='pan1')
    parser.add_argument('-pan2', default=0, type=float, help="Pan of 2nd source.", dest='pan2')
    parser.add_argument('-pan3', default=0, type=float, help="Pan of 3rd source.", dest='pan3')
    parser.add_argument('-pan4', default=0, type=float, help="Pan of 4th source.", dest='pan4')
    parser.add_argument('-pan5', default=0, type=float, help="Pan of 5th source.", dest='pan5')
    parser.add_argument('-pan6', default=0, type=float, help="Pan of 6th source.", dest='pan6')
    parser.add_argument('-pan7', default=0, type=float, help="Pan of 7th source.", dest='pan7')
    parser.add_argument('-pan8', default=0, type=float, help="Pan of 8th source.", dest='pan8')
    
    options = parser.parse_args()
    print(options)

    main(options.filename, options.j, options.epoch,
         options.v, options.g, options.p, options.t, options.s, options.d,
         options.pan1, options.pan2, options.pan3, options.pan4, 
         options.pan5, options.pan6, options.pan7, options.pan8)
    
    