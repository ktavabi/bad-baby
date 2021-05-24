#!/usr/bin/env python

"""
Todo:
- Add plotting as a function of time for each contrast
"""

import itertools
import os
import os.path as op
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import janitor  # noqa
import mne
from mne import read_epochs
from mne.decoding import (
    SlidingEstimator, cross_val_multiscore, LinearModel, Scaler, Vectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import mnefun

static = op.join(Path(__file__).parents[1], "static")
params = mnefun.read_params(
    op.join(Path(__file__).parents[1], "processing", "badbaby.yml")
)
meg_features = (
    pd.read_excel(op.join(static, "meg_covariates.xlsx"), sheet_name="mmn")
    .clean_names()
    .rename_columns({"subjid": "id"})
    .filter_on("complete == 1", complement=False)
)
subjects = sorted(f"bad_{id_}" for id_ in meg_features["id"])

workdir = op.join(params.work_dir, '..', 'badbaby_data')
out_dir = op.join(op.dirname(__file__), 'results')
os.makedirs(out_dir, exist_ok=True)
intervals = ['First ⅓', 'Second ⅓', 'Third ⅓', 'All']
plt.style.use('ggplot')
ages = [2, 6]
solver = 'liblinear'
n_jobs = 4
lp = params.lp_cut
win = (0.235, 0.53)
n_splits = 5  # how many folds to use for cross-validation
seed = np.random.RandomState(42)
joint_kwargs = dict(ts_args=dict(gfp=True, time_unit='s'),
                    topomap_args=dict(sensors=False, time_unit='s'))
conditions = ['std', 'ba', 'wa']
combine = False
events = list(itertools.combinations(conditions, 2))
aucs = np.zeros((len(subjects), len(events), len(intervals)))
for si, subject in enumerate(subjects):
    fname = op.join(out_dir, f'INTERVALS_{subject}.h5')
    if op.isfile(fname):
        data = mne.externals.h5io.read_hdf5(fname)
        assert events == data['events']
        assert intervals == data['intervals']
        aucs[si] = data['auc']
        continue
    print('Fitting estimator for %s: ' % subject)
    ep_fname = op.join(workdir, subject, 'epochs',
                        'All_%d-sss_%s-epo.fif' % (lp, subject))
    epochs = read_epochs(ep_fname)  # XXX apply baseline?
    epochs = epochs['std', 'ba', 'wa']
    epochs.pick_types(meg=True)  # get rid of trigger channels
    epochs.drop_bad()
    epochs.equalize_event_counts(epochs.event_id.keys())
    epochs.filter(None, 10).decimate(6)  # lowpass and decimate
    epochs.crop(*win)
    for ci, cs in enumerate(events):
        for ii, interval in enumerate(intervals):
            eps = epochs[cs]
            if 'First' in interval:
                sl = slice(None, len(eps) // 3)
            elif 'Second' in interval:
                sl = slice(len(eps) // 3, 2 * len(eps) // 3)
            elif 'Third' in interval:
                sl = slice(2 * len(eps) // 3, None)
            else:
                assert interval == 'All'
                sl = slice(None)
            eps = eps[sl]
            info = eps.info
            time = eps.times
            ix = eps.time_as_index(win[0])[0], eps.time_as_index(win[1])[0]
            c1, c2 = list(eps.event_id.keys())
            clf = make_pipeline(Scaler(eps.info),
                                Vectorizer(),
                                PCA(0.9999),
                                LinearModel(
                                    LogisticRegression(solver=solver,
                                                       penalty='l1',
                                                       max_iter=1000,
                                                       multi_class='auto',
                                                       random_state=seed)))
            time_decode = SlidingEstimator(clf, n_jobs=n_jobs,
                                           scoring='roc_auc',
                                           verbose=False)
            # K-fold cross-validation with ROC area under curve score
            if subject[4:] in ('124a', '213', '301a') and interval != 'All':
                use_splits = 3
            else:
                use_splits = n_splits
            cv = StratifiedKFold(n_splits=use_splits, shuffle=True,
                                 random_state=seed)
            # Get the data and label
            X = eps.get_data()
            y = eps.events[:, -1]
            # AUC b/c chance level same regardless of the class balance
            score = np.mean(cross_val_multiscore(time_decode, X=X, y=y,
                                                 cv=cv, n_jobs=n_jobs),
                            axis=0)
            aucs[si, ci, ii] = score.mean()
            print("  %s vs. %s mean AUC: %.3f on %d epochs" % (
                c1, c2, score.mean(), len(eps)))
    mne.externals.h5io.write_hdf5(
        fname, dict(auc=aucs[si], events=events, intervals=intervals))

assert (meg_features['id'] == [subj[4:] for subj in subjects]).all()
ages = np.array(meg_features['age'])
age_bounds = [[40, 80], [105, 150], [175, 210], [40, 210]]
fig, axes = plt.subplots(
    len(events), len(age_bounds), constrained_layout=True,
    figsize=(7, 5), squeeze=False, sharex=True, sharey=True)
for ei, event in enumerate(events):
    used = np.zeros(len(subjects), bool)
    for ai, age in enumerate(age_bounds):
        ax = axes[ei, ai]
        ylabel = '-'.join(event)
        if ai == 0:
            ax.set(ylabel=f'{ylabel} AUC')
        mask = (ages >= age[0]) & (ages <= age[1])
        if ai != 3:
            assert not used[mask].any()
        else:
            assert used[mask].all() and mask.all()
        used[mask] = True
        data = aucs[mask][:, ei]
        x = np.arange(len(intervals))
        m = np.mean(data, axis=0)
        s = np.std(data) / np.sqrt(len(data))
        ax.bar(x, m, yerr=s, ecolor='k', facecolor='none', edgecolor='k', lw=1,
               zorder=5)
        ax.plot(x, data.T, color='k', alpha=0.05, zorder=4)
        ax.set_xticks(x)
        if ei == len(events) - 1:
            ax.set_xticklabels(
                intervals, rotation=45, rotation_mode='anchor', ha='right')
            ax.set(xlabel=f'Interval')
        else:
            ax.set_xticklabels([''] * len(x))
        if ei == 0:
            ax.set(title=f'Age {age[0]}-{age[1]}')
        ax.set(ylim=[0.25, 0.75])
        ax.axhline(0.5, color='k', lw=1, ls=':')
    assert used.all()
fig.savefig(op.join(out_dir, 'decoding_epochs.png'))
