#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:54:15 2020

@author: pettetmw
"""

#%%

import os
from glob import glob
import numpy as np
import pandas as pd
from itertools import product as iprod
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist
import mne
from mne.externals.h5io import write_hdf5

dict2df = lambda d : pd.DataFrame.from_records(iprod(*d.values()),columns=d.keys())


#%%
def makeTFRfromEpo(asp):
    """
    makeTFRfromEpo 

    Parameters
    ----------
    asp : TYPE string
        DESCRIPTION. a subject's path

    Returns
    -------
    tfr_epo, pow_epo, pow_evo, itc_evo, tcirc_evo

    """
    epo = mne.read_epochs(asp)
    tNTrl, tNCh, tNS = epo.get_data().shape
    
    # Epoch is -500ms - 1000ms, baseline is -500ms - 0ms i.e. work on nsamples between epochs.tmin to epochs.tmax
    # stimulus onset is eps.time_as_index(eps.time == 0) sample point for stimulus onset.
    
    """
    Not quite:
        epo.get_data().shape == (120, 306, 781)
        [ epo.tmin, epo.tmax ] == [ -0.2, 1.1 ]
        epo.time_as_index(0.0) == 120
        epo.info['sfreq'] == 600.0
    """
    
    # (i) BP 28-56 Hz with np.logspace(*np.log10([28, 56]), num=16) as input arg to TFR function (morlet / multitaper)
    
    # freqs = np.logspace(*np.log10([28, 56]), num=16)  # log-spaced passbands
    
    # alternative version for linear spacing between freqs:
    # freqs = np.fft.fftfreq(tNS,1.0/epo.info['sfreq']);
    
    # log spaced passpands around explicit 40.0 Hz
    # freqs = np.concatenate( (
    #     np.logspace(*np.log10([28, 39]), num=7),
    #     np.array((40.0,)),
    #     np.logspace(*np.log10([41, 56]), num=7) ) )
    
    freqs = np.arange(34.0,48.0,2.0)
    
    # Coerce n_cycles to scale with freq to s.t. all
    # passbands have same time window duration
    n_cycles = freqs / 2.  
    
    # (iiia) complex ITC epoch object per subject with data array (nchs x nfreqs x nsamples)
    # (iiib) complex ITC evoked object per subject with array (nfrqs x nsamples) <--> GFP?
    
    """
    given following excerpt from from tfr_morlet() docstring:
    
        return_itc : bool, default True
            Return inter-trial coherence (ITC) as well as averaged power.
            Must be ``False`` for evoked data.
        average : bool, default True
            If True average across Epochs.
        output : str
            Can be "power" (default) or "complex". If "complex", then
            average must be False.
    
        Returns
        -------
        power : AverageTFR | EpochsTFR
            The averaged or single-trial power.
        itc : AverageTFR | EpochsTFR
            The inter-trial coherence (ITC). Only returned if return_itc
            is True.
    
    
    and the following undocumented restriction:
    ValueError: Inter-trial coherence is not supported with average=False
    
    (I guess that should be self-evident since "inter-trial" implies a summarization
     over trials, but if that's the case, what's up with the "EpochsTFR" return
     value option in the docstring above for "itc"?)
    
    So we can compute the following quantities:
    """
    
    # complex EpochsTFR
    tfr_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, average=False, output='complex')
    
    
    # power EpochsTFR
    pow_epo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=False, average=False, output='power')
    
    # power and ITC AverageTFR
    pow_evo, itc_evo = mne.time_frequency.tfr_morlet( epo, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, average=True, output='power')
    
    # QC:
    # tfr_epo.data.shape == (120, 306, 16, 781) 
    # pow_epo.data.shape == (120, 306, 16, 781)
    # pow_evo.data.shape == (306, 16, 781)
    # itc_evo.data.shape == (306, 16, 781)
    
    # QC: the following should have a band at 40 Hz from 0 to 1000 ish
    
    #pow_evo.plot_joint()
    #itc_evo.plot_joint()
    
    #mne.time_frequency.AverageTFR.plot_joint()
    
    # (vi) Yield (e.g. Tcirc) should be array like per subject (nfreqs x nsamples) prior to group level testing
    #
    tNTrl, tNCh, tNFrq, tNS = tfr_epo.data.shape
    
    tMYFFT = np.mean( tfr_epo.data, axis=0 )
    # compute the mean of the variances along real and imaginary axis
    tYFFTV = np.mean( np.stack( ( np.var( np.real(tfr_epo.data), 0 ), np.var( np.imag(tfr_epo.data), 0 ) ) ), 0 )
    #tYFFTV = np.var( abs(tYFFT), 0 )
    numerator = abs(tMYFFT);
    denominator = np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = abs(tMYFFT) / np.sqrt( tYFFTV / ( tNTrl - 1 ) )
    #tcirc = numerator / denominator
    tcirc = (numerator / denominator)**2
    tcirc_evo = itc_evo.copy()
    tcirc_evo.data = tcirc
    
    # note that AverageTFR.plot(...'combine=mean') implies mean over chan picks, s.t.
    # heat map is (nfreqs x nsamples), ready for (viia,b) below
    
    return [ 'tfr-epo', 'pow-epo', 'pow-evo', 'itc-evo', 't2c-evo'], \
        [ tfr_epo, pow_epo, pow_evo, itc_evo, tcirc_evo ]
    # QC:
    # tfr_epo.data.shape == (120, 306, 16, 781) # complex-valued tf response
    # pow_epo.data.shape == (120, 306, 16, 781) # power of  tfr_epo
    # pow_evo.data.shape == (306, 16, 781)
    # itc_evo.data.shape == (306, 16, 781)
    # tcirc_evo.data.shape == (306, 16, 781)
    
    
 
#%%(iv)
# loop over subjects to create files
epop = '/mnt/scratch/badbaby/tone/' # epoch file path
tfrp ='/mnt/scratch/badbaby/pipe/TFRs/'

# list comprehension to gather epoch file paths from group "a" and "b"
epops = [ sorted(glob(epop + 'bad*'+g+'/epochs/*epo.fif')) for g in ['a', 'b'] ] # get the existing a's and b's

# cull "a"'s without 2nd visits...

## for comparison, the adult pilot
#tPaths = sorted(glob(epop + 'bad_000/epochs/bad_000_tone_epo.fif')) # bad_000 is Adult pilot (E Larson)

#epops = [ epops[0][:3], epops[1][:3] ] # crop for quickie test

for g in epops: # each group
    for sidp in g: # each subject id path
        sid = os.path.basename(sidp)[:-8] # subject id
        tfr = makeTFRfromEpo(sidp) # a 2-tuple: ( [names], [tfr object variables] )
        for iv,vn in enumerate(tfr[0]): # tfr[0] is the list of tfr variable names
            tfrvp = tfrp + sid + '_' + vn + '_tfr.h5' # tfr variable path
            tfr[1][iv].save(tfrvp,overwrite=True) # tfr[1] is the list of tfr object variables



#%%(iv,cont...)
# loop over subjects to aggregate group summary stats below.

# # redefine variables here if needed to use different location for stored data files...
# epop = '/mnt/scratch/badbaby/tone/' # epoch file path
# tfrp ='/mnt/scratch/badbaby/pipe/TFRs/'

# list comprehension to gather tfr evo file paths from group "a" and "b"
#%% first for Power...

# list comprehension to gather tfr evo file paths from group "a" and "b"
tfrevops = [ sorted(glob(tfrp + 'All*'+g+'_pow-evo_tfr.h5')) for g in ['a', 'b'] ] # one list each for the existing a's and b's
tfrevos = [ [ mne.time_frequency.read_tfrs(sidp)[0] for sidp in g ] for g in tfrevops ] # read in each sid for each group

tfrms = [ np.mean( np.stack( [ s.data for s in g ] ), 0 ) for g in tfrevos ] # means over sids for each group

glabs = ['2mo', '6mo']
for ig,g in enumerate(tfrms):
    tfr = tfrevos[0][0].copy()
    tfr.data = g
    tfr.plot(picks='grad',combine='mean',title=glabs[ig]+', Power over gradiometers', vmin=0.0, vmax=3);
    tfr.plot_topomap( ch_type='grad', fmin=38.0,fmax=42.0,tmin=0.5,tmax=1.0,sphere=1.0, vmin=0.0, vmax=3);

#%% ibid., for ITC...
tfrevops = [ sorted(glob(tfrp + 'All*'+g+'_itc-evo_tfr.h5')) for g in ['a', 'b'] ] # one list each for the existing a's and b's
tfrevos = [ [ mne.time_frequency.read_tfrs(sidp)[0] for sidp in g ] for g in tfrevops ] # read in each sid for each group
tfrms = [ np.mean( np.stack( [ s.data for s in g ] ), 0 ) for g in tfrevos ] # means over sids for each group

glabs = ['2mo', '6mo']
for ig,g in enumerate(tfrms):
    tfr = tfrevos[0][0].copy()
    tfr.data = g
    tfr.plot(picks='grad',combine='mean',title=glabs[ig]+', ITC over gradiometers', vmin=0.0, vmax=0.1);
    tfr.plot_topomap( ch_type='grad', fmin=38.0,fmax=42.0,tmin=0.5,tmax=1.0,sphere=1.0, vmin=0.0, vmax=0.1);

#%% ibid., for T2circ...

# list comprehension to gather tfr evo file paths from group "a" and "b"
tfrevops = [ sorted(glob(tfrp + 'All*'+g+'_tcirc-evo_tfr.h5')) for g in ['a', 'b'] ] # one list each for the existing a's and b's
tfrevos = [ [ mne.time_frequency.read_tfrs(sidp)[0] for sidp in g ] for g in tfrevops ] # read in each sid for each group
tfrms = [ np.mean( np.stack( [ s.data for s in g ] ), 0 ) for g in tfrevos ] # means over sids for each group

glabs = ['2mo', '6mo']
for ig,g in enumerate(tfrms):
    tfr = tfrevos[0][0].copy()
    tfr.data = g
    tfr.plot(picks='grad',combine='mean',title=glabs[ig]+', T2circ over gradiometers', vmin=0.0, vmax=3);
    tfr.plot_topomap( ch_type='grad', fmin=38.0,fmax=42.0,tmin=0.5,tmax=1.0,sphere=1.0, vmin=0.0, vmax=3);



# PICK UP HERE:
    
    # plot random subsample of evo arrays as 2d image plots in subplots
    # a vs b, sbj-by-sbj subplots

#%% (viia) Grand averaged TFR plots for band passed ITC or phase locking for
#          each age group. Plots of phase coh over time.

# do plot_joint() on group result
#line plot of power over time for  39,40,41
# maybe use seaborn



#%% (viib) Group level analysis should be done on array (nsubjs x nfreqs x nsamples) per age.

# time-freq descriptive variable: mean over grad and atpicks for each afpicks
def tfrdv(atfr,atpicks,afpicks) : # returns tfr descriptive value, .shape == (3,), one for each fpick
    return np.mean(np.mean(atfr.copy().pick('grad').data[:,:,atpicks],-1),0)[afpicks]

age2gid = { '2mo':'a', '6mo':'b' } # age-to-groupID ('a' and 'b' are parts of file names)
tfr_vnms = [ 'pow', 'itc', 't2c' ] # time-freq response variable names (also part of file name)

def tfr_stats(age,tfr): # time-freq resp stat for given age and tfr type
    # hereafter, "sid" is subject id
    sidps = sorted(glob(tfrp+'All*{}_{}-evo_tfr.h5'.format(age2gid[age],tfr))) # sid paths, ...
        # e.g., /mnt/scratch/badbaby/pipe/TFRs/All_100-sss_bad_921b_t2c-evo_tfr.h5
    # sidps = sidps[:4] # shorty version for quick testing
    sids = [ os.path.basename(sidp)[-19:-16] for sidp in sidps ] # get id from full path
    tfrevos = [ mne.time_frequency.read_tfrs(sidp)[0] for sidp in sidps ] # tfr evo objects read from each path
    fpicks = [f >= 38.0 and f <= 42.0 for f in tfrevos[0].freqs] # frequency picks QC: should be [38.0, 40.0, 42.0]
    tpicks = [ t >= .2 and t < .8 for t in tfrevos[0].times ] # time picks
    freqs = tfrevos[0].freqs[fpicks] # the picked frequencies, for labeling
    tfrdvs = np.concatenate( [ tfrdv(at,tpicks,fpicks) for at in tfrevos ] )
    # now build pandas DataFrame with categorical key columns
    df = dict2df( { 'age':[age], 'sid':sids, 'tfrvar':[tfr], 'freqs':freqs } )
    df['tfrval'] = tfrdvs
    return df
    
pdfcat = lambda adf : pd.concat(adf,ignore_index=True)
ttab = pdfcat( [ pdfcat( [ tfr_stats(age,tfr) for tfr in tfr_vnms ] ) for age in age2gid.keys() ] )
ttab.to_csv('tfr_stats.csv')

#%% (v) Check whether (a) t-circ == Hoetellings and (b) whether values are F distributed



#%% Hiding place for single-shot QC code fragments.

# This section still needs revision to read the saved tfr descriptives (instead
# of calling makeTFRfromEpo).

# #%% (ii) Single subject (adult test dataset) full length epochs for nTrials.

# # sid = '000' # subject id for adult
# # sp = '../tone/bad_'+sid+'/epochs/bad_'+sid+'_tone_epo.fif' # subject path

# sid = '116b' # subject id for infant with robust ASSR
# sp = '../tone/bad_'+sid+'/epochs/All_100-sss_bad_'+sid+'-epo.fif'

# #tfr_epo, pow_epo, pow_evo, itc_evo, tcirc_evo = makeTFRfromEpo(sp)
# tfrt = makeTFRfromEpo(sp) # as tuple

# #%%

# # QC visualization of single subject
# tfrd = dict( zip(*tfrt) )

# # plot_joint works for TF image, and correctly sets target for topo image,
# # but mangles topo images
# #tcirc_evo.plot_joint(picks='meg',combine='mean',timefreqs={(0.6,40): (0.25,5)});

# # ITC TF plot and topomap
# tfrd['itc-evo'].plot(picks='grad',combine='mean',title='Bad'+sid+', ITC over gradiometers');
# tfrd['itc-evo'].plot_topomap( ch_type='grad', fmin=38.0,fmax=42.0,tmin=0.5,tmax=1.0,sphere=1.0);

# # T2circ TF plot and topomap
# tfrd['tcirc-evo'].plot(picks='grad',combine='mean',title='Bad'+sid+', T2circ over gradiometers');
# tfrd['tcirc-evo'].plot_topomap( ch_type='grad', fmin=38.0,fmax=42.0,tmin=0.5,tmax=1.0,sphere=1.0);
    

