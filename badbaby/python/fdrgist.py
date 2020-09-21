#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:40:04 2020

@author: pettetmw
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 11:33:33 2020

@author: pettetmw
"""
#%%

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
import mne
from mne.externals.h5io import write_hdf5
#import mnefun
from fake import *

#%%
# the following function is called from bash command line
# to populate subdirs tcircevo and pcircevo
# with EvokedArray files containing t-circ values for each baby


def TCirc(atcevop,aepop,tmin=None,tmax=None,fundfreqhz=None):
    # create a tcirc evoked file (atcevop) from epoched file (aepop)
    # note that np.fft.fft "axis" parameter defaults to -1
    
    # Be careful about trailing samples when converting from time to array
    # index values

    #plt.plot(np.mean(tY,axis=-1),'r-')
    
    epo = mne.read_epochs(aepop)
    
    info = epo.info
    sfreq = info['sfreq'] # to compute location of tmin,tmax
    imn = int( sfreq * tmin )
    imx = int( sfreq * tmax )
    
    tY = epo.get_data()[ :, :, imn:imx ] # remove odd sample (make this conditional)

    tNTrl, tNCh, tNS = tY.shape # number of trials, channels, and time samples
    
    tMYFFT = np.fft.fft( np.mean( tY, axis=0 ) ) / tNS # FFT of mean over trials of tY, Chan-by-Freq
    tYFFT = np.fft.fft( tY ) / tNS # FFT of tY , Trials-by-Chan-by-Freq
    
    # compute the mean of the variances along real and imaginary axis
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tYFFT), 0 ), np.var( np.imag(tYFFT), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    numerator = abs(tMYFFT);
    denominator = np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = numerator / denominator
    tcirc = (numerator / denominator)**2
    pcirc = 1 - f_dist(2,2*(tNTrl-1)).cdf(tcirc)
    # pmask, pcirc = mne.stats.fdr_correction( pcirc )
   
    info['sfreq']=0.5
    mne.EvokedArray( tcirc, info ).save( atcevop )
    mne.EvokedArray( pcirc, info ).save( atcevop.replace('tcircevo','pcircevo') )
    return tcirc 


#%%

# the following statments accumulate the contents of those files:
# "lnx" is a wrapper for linux subprocess call with extended globbing enabled (see fake.py).
# Below, it invokes the "ls" command to find the files in epochs, tcircevo, and pcircevo

# tPaths = [ sorted(lnx('ls -1 ../tone/*/epochs/*%s-epo.fif' % g)) for g in ['a', 'b'] ]
# evos = [ [ mne.read_epochs(p).average() for p in g ] for g in tPaths ]
# evos = dict( zip( agegroups, evos ))
# tNDOFs = dict( zip( agegroups, [ [ e.nave for e in evos[ag] ] for ag in agegroups ] ) )

agegroups = ['2mo','6mo']

tPaths = [ sorted(lnx('ls -1 tcircevo/*%s-ave.fif' % g)) for g in ['a', 'b'] ]
tcircevos = dict( zip( agegroups, [ [ mne.read_evokeds(p)[0].crop(0,50) for p in g ] for g in tPaths ] ) )

tPaths = [ sorted(lnx('ls -1 pcircevo/*%s-ave.fif' % g)) for g in ['a', 'b'] ]
pcircevos = dict( zip( agegroups, [ [ mne.read_evokeds(p)[0].crop(0,50) for p in g ] for g in tPaths ] ) )

#%%

for ag in agegroups :
    
    # pcircave = mne.grand_average( pcircevos[ag], pcircevos[ag][0].info )
    # print(np.min(pcircave.data))
    
    tAllPVals = np.stack( [ p.data for p in pcircevos[ag] ], 2 )
    fdrpmask, fdrpcirc = mne.stats.fdr_correction( tAllPVals )
    
    
    tEvoP = mne.EvokedArray( np.mean(tAllPVals,2), info=pcircevos[ag][0].info )
    tEvoP.plot_topomap(title=ag+' uncorrected p-values',
                       times=[2.0,4.0,6.0,38.0,40.0,42.0], contours=[0.05,0.2,0.4,0.6,0.8],
                       vmax=0.99, vmin=0.0, cbar_fmt='%0.1f', scalings=1);
    
    tEvoP = mne.EvokedArray( np.mean(fdrpcirc,2), info=pcircevos[ag][0].info )
    tEvoP.plot_topomap(title=ag+' FDR p-values',
                       times=[2.0,4.0,6.0,38.0,40.0,42.0], mask=np.any(fdrpmask,2), contours=[0.05,0.2,0.4,0.6,0.8],
                       vmax=0.99, vmin=0.0, cbar_fmt='%0.1f', scalings=1);


##%%
    tSomePVals = tAllPVals[:,[0,1,2,20],:]
    fdrpmask, fdrpcirc = mne.stats.fdr_correction( tSomePVals )
    
    
    tEvoP = mne.EvokedArray( np.mean(tSomePVals,2), info=pcircevos[ag][0].info )
    tEvoP.plot_topomap(title=ag+' uncorrected p-values',
                       times=[0.0,2.0,4.0,6.0], contours=[0.05,0.2,0.4,0.6,0.8],
                       vmax=0.99, vmin=0.0, cbar_fmt='%0.1f', scalings=1);
    
    tEvoP = mne.EvokedArray( np.mean(fdrpcirc,2), info=pcircevos[ag][0].info )
    tEvoP.plot_topomap(title=ag+' FDR p-values',
                       times=[0.0,2.0,4.0,6.0], mask=np.any(fdrpmask,2), contours=[0.05,0.2,0.4,0.6,0.8],
                       vmax=0.99, vmin=0.0, cbar_fmt='%0.1f', scalings=1);




















