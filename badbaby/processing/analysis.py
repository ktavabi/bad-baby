#!/usr/bin/env python
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from os import path as op

import janitor  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_flavor as pf  # noqa
import seaborn as sns
from mne import combine_evoked, read_evokeds
from mne.utils.numerics import grand_average
from mnefun import read_params as rp
from pandas_profiling import ProfileReport

# %% [markdown]
# ### Sample charactersitics & findings
# -  Strong correlations betweeen SES and parental education levels.
# TODO Get REFS  for impact of parental education on SES.
#

# %%
df = (
    pd.read_excel(op.join(static, "meg_covariates.xlsx"), sheet_name="mmn")
    .clean_names()
    .select_columns(
        [
            "subjid",
            "ses",
            "age",
            "gender",
            "headsize",
            "maternaledu",
            "paternaledu",
            "maternalethno",
            "paternalethno",
        ]
    )
    .dropna(inplace=False)
    .rename_columns({"subjid": "sid"})
    .encode_categorical(columns=["gender", "maternalethno", "paternalethno"])
    .bin_numeric(
        from_column_name="maternaledu",
        to_column_name="Education (yrs)",
        num_bins=3,
        labels=["9-12", "13-17", "18-21"],
    )
    .groupby_agg(
        by="gender",
        new_column_name="grouping",
        agg_column_name="age",
        agg="mean",
        dropna=True,
    )
)
df['sid'] = ['bad_%s' % xx for xx in df['sid']]

print(df.groupby("gender").describe(percentiles=[0.5]))

# %%
cdi = (
    pd.read_excel(op.join(static, "behavioral_data.xlsx"), sheet_name="Data")
    .clean_names()
    .rename_columns({"subjid": "sid"})
    .drop(
        columns=[
            "dob",
            "gender",
            "language",
            "cdiform",
            "examdate",
            "vocabper",
            "howuse",
            "upstper",
            "ufutper",
            "umisper",
            "ucmpper",
            "uposper",
            "wordend",
            "plurper",
            "possper",
            "ingper",
            "edper",
            "irwords",
            "irwdper",
            "ogwords",
            "combine",
            "combper",
            "cplxper",
        ],
        axis=1,
    )
)
cdi['sid'] = cdi["sid"].str.lower()
cdi.info
grouped = cdi.groupby("cdiage")
print(grouped.describe())

# %%
covars = df.merge(cdi, on='sid', sort=True)
covars.groupby("cdiage").describe(percentiles=[0.5])

# %%
sns.set_theme(style="ticks", color_codes=True)
# plot distributions of SES amongst genders as a function of mother's years-edu
sns.catplot(
    x="Education (yrs)", y="ses", hue="gender", kind="bar", data=covars
)
sns.catplot(
    y="maternalethno", hue="gender", kind="count", data=covars
)  # so much for equity & diversity
sns.lmplot(y="ses", x="maternaledu", data=covars, x_estimator=np.mean)
sns.lmplot(y="age", x="headsize", data=covars, x_estimator=np.mean)

# %%
profile = ProfileReport(covars, title="features profile").to_widgets()

# %% [markdown]
# #### Grand averaging

# %%
prms = rp(
    op.join("/home/ktavabi/Github/badbaby/badbaby", "processing", "badbaby.yml")
)
wrkDir = "/media/ktavabi/ALAYA/data/ilabs/badbaby"
lp = prms.lp_cut


# %%
# averaging gist c/o Larson
conditions = ["standard", "deviant"]
evoked_dict = dict((key, list()) for key in conditions)
for condition in conditions:
    for subject in prms.subjects:
        fname = op.join(
            wrkDir, subject, "inverse", f"Oddball_{lp}-sss_eq_{subject}-ave.fif"
        )
        evoked_dict[condition].append(read_evokeds(fname, condition))
evoked = combine_evoked(
    [grand_average(evoked_dict[condition]) for condition in conditions],
    weights=[-1, 1],
)
evoked.pick_types(meg=True)
evoked.plot_joint()


# %%
events = {"standard": [], "deviant": []}
for condition in events.keys():
    evs = list()
    for si, subj in enumerate(prms.subjects):
        print(" %s" % subj)
        filename = op.join(
            wrkDir, subj, "inverse", "Oddball_%d-sss_eq_%s-ave.fif" % (lp, subj)
        )
        evs.append(
            read_evokeds(filename, condition=condition, baseline=(None, 0))
        )
    # do grand averaging
    print("  Doing %s averaging." % condition)
    events[condition].append(grand_average(evs))


# %%
import datetime
import itertools
import os.path as op
import re

import numpy as np
import pandas as pd
import xarray as xr

# %%
plt.style.use("ggplot")

date = datetime.datetime.today()
date = "{:%m%d%Y}".format(date)
analysese = ["Individual", "Oddball"]
ages = [2, 6]
regex = r"[0-9]+"
solver = "lbfgs"

# %%

# TODO Wrangle MEG decoding metrics and demographic, SES, CDI response measures into TIDY format.
dfs = list()
for iii, analysis in enumerate(analysese):
    print("Reading data for %s analysis... " % analysis)
    if iii == 0:
        conditions = ["standard", "ba", "wa"]
    else:
        conditions = ["standard", "deviant"]
    combos = list(itertools.combinations(conditions, 2))
    fi_in = op.join(
        defaults.datadir, "AUC_%d_%s_%s.nc" % (lp, solver, analysis)
    )
    ds = xr.open_dataarray(fi_in)
    dfs.append(ds.to_dataframe(name="AUC").reset_index())
df = pd.concat(dfs, axis=0, verify_integrity=True, ignore_index=True)
df.rename(columns={"subject": "megId"}, inplace=True)
Ds = df.merge(covars, on="megId", validate="m:m")
mapping = {
    "standard-ba": "plosive",
    "standard-wa": "aspirative",
    "standard-deviant": "mmn",
    "ba-wa": "deviant",
}
Ds.replace(mapping, inplace=True)
Ds["vocab-asin"] = np.arcsin(
    np.sqrt(Ds.vocab.values / Ds.vocab.values.max())
)  # noqa
Ds["m3l-asin"] = np.arcsin(np.sqrt(Ds.m3l.values / Ds.m3l.values.max()))
Ds.info()
Ds.to_csv(op.join(wrkDir, "cdi-meg_%d_%s_tidy_%s.csv" % (lp, solver, date)))
