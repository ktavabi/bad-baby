#!/usr/bin/env python

# Authors:  Kambiz Tavabi <ktavabi@gmail.com>
#           Mark Pettet <pettetmw@gmail.com>
"""
Script to write to disk individual subject trial-average ITC to MNEFUN tree as 
MNE-python evokeds object.
"""

# %%
import mne
import numpy as np
from mne.time_frequency import tfr_morlet

# %%
freqs = np.arange(34.0, 48.0, 2.0)
n_cycles = freqs / 2.  # Coerce n_cycles to same time window duration

# Compute TFRs as individual subject evoked objects.

# #TODO MNEFUN compliance @pettetmw
# #TODO BIDS compliance @ktavabi
foo = list()  # container for trial spectral data
for ss in picks:  # input list of subject ids read from excel files in static assets directory.
    eps = mne.read_epochs(eps_fname)
    for output in ['complex', 'power']:
        foo.append(tfr_morlet(eps, freqs=freqs, n_cycles=n_cycles, 
        use_fft=True, return_itc=False, average=False, output=output))

    # Evoked power and ITC
    pow, itc = tfr_morlet(eps, freqs=freqs, n_cycles=n_cycles, use_fft=True,
    return_itc=True, average=True, output='power')

    mofft = np.mean(itc.data, axis=0)  # Mean of FFT
    # compute the mean of the variances along real and imaginary axes
    vofft = np.mean(np.stack((np.var(np.real(tfr_epo.data), 0),np.var(np.imag(tfr_epo.data), 0))), 0)
    numerator = abs(mofft)
    denominator = np.sqrt(vofft / (tNTrl - 1))
    tcirc = (numerator / denominator)**2
    tcirc_evo = itc.copy()  # TODO @pettetmw use mne make info and clobber data array into clean evoked object
    tcirc_evo.data = tcirc  # TODO @pettetmw save evoked in MNEFUN tree per ID
