#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:13:14 2020

@author: pettetmw
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
import mne
from mne.externals.h5io import write_hdf5
from mne.stats import fdr_correction as fdr
#import mnefun
from fake import *

PR = lambda x : print(np.round(x,3))

#%%
# tF = f(2,40)
# tT2 = np.sort( tF.rvs(30) );
# PR(tT2)
# PR(tF.cdf(tT2))
# PR(tF.sf(tT2))
# PR(fdr(tF.sf(tT2)))

#%%
def TCirc(atcevop,aepop,tmin=None,tmax=None,fundfreqhz=None):
    # create a tcirc evoked file (atcevop) from epoched file (aepop)
    # note that np.fft.fft "axis" parameter defaults to -1
    
    # Be careful about trailing samples when converting from time to array
    # index values

    #plt.plot(np.mean(tY,axis=-1),'r-')
    
    epo = mne.read_epochs(aepop)
    
    info = epo.info
    sfreq = info['sfreq'] # to compute location of tmin,tmax
    # imn = int( sfreq * tmin )
    # imx = int( sfreq * tmax )
    
    # tY = epo.get_data()[ :, :, imn:imx ] # remove odd sample (make this conditional)
    
    #epo = epo.crop(tmin,tmax)
    # epo = epo.crop(0.5,1.0);
    tY = epo.get_data();

    tNTrl, tNCh, tNS = tY.shape # number of trials, channels, and time samples
    
    
    tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    tYFFT = np.fft.fft( tY ) / tNS # FFT of tY , Trials-by-Chan-by-Freq
    
    
    
    
    freqs = np.fft.fftfreq(tNS) # np.logspace(*np.log10([28, 56]), num=16)  # band-passing to focus on 40Hz response
    freqs = freqs[18:23]
    n_cycles = freqs / 2.  # n cycle per frequency
    # morlet = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
    #                     return_itc=True, average=False, output='complex')
    
    morlet, itc = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, average=True, output='power')
    

    
    
    # compute the mean of the variances along real and imaginary axis
    tYFFTV = np.var( np.real(tYFFT), 0 ) + np.var( np.imag(tYFFT), 0 )
    numerator = abs(tMYFFT)**2
    denominator = np.var( np.real(tYFFT), 0 ) + np.var( np.imag(tYFFT), 0 )
    tcirc = numerator / denominator
    pcirc = 1 - f_dist(2,2*(tNTrl-1)).cdf(tcirc)
   
    # info['sfreq']=0.5
    # mne.EvokedArray( tcirc, info ).save( atcevop )
    # mne.EvokedArray( pcirc, info ).save( atcevop.replace('tcircevo','pcircevo') )
    #morlet.save(atcevop.replace('tcircevo','morlet')) # ...-tfr.h5
    return tcirc, morlet 

#%%
# sid = '116b'
# p = '../tone/bad_'+sid+'/epochs/All_100-sss_bad_'+sid+'-epo.fif'
# n = len(mne.read_epochs(p).events)
# ddof = 2*(n-1)
# p = 'tcircevo/All_100-sss_bad_'+sid+'-ave.fif' # 110 trials
# tcircs = mne.read_evokeds(p)[0].data

# obsTCs = np.flip(np.sort(tcircs[:,18:23],0))
# rndTCs = np.flip(np.sort( f_dist(2,ddof).rvs(306) ))
# rndPs = f_dist(2,ddof).sf(rndTCs)
# obsPs = f_dist(2,ddof).sf(obsTCs)

# plt.figure(); plt.title(str(sid)+', T2circ scores by rank (over channels)')
# plt.plot(obsTCs)
# plt.plot(rndTCs,'--')
# plt.legend(['36Hz', '38Hz', '40Hz', '42Hz', '44Hz', 'F(2,'+str(ddof)+')'])
# plt.figure(); plt.title(str(sid)+', P values by rank (over channels)')
# plt.plot(obsPs)
# plt.plot(rndPs,'--')
# plt.legend(['36Hz', '38Hz', '40Hz', '42Hz', '44Hz', 'F(2,'+str(ddof)+')'])

#%%
sid = '000'
p = '../tone/bad_000/epochs/bad_000_tone_epo.fif'

# lnx('ls -l '+p)

# makeTcircevoFromEpo=lambda t,p: TCirc(t,p,tmin=0.5,tmax=1.0)

tcirc, morlet = TCirc( '', p, tmin=0.5, tmax=1.0 )










