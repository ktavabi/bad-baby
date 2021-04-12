#!/usr/bin/env python
"""MNEFUN preprocessing pipeline.

Subjects whose data were not on the server under their given names with all
runs were 104, 114 , 120b, 127b, 130a, 203, 209a, 213 , 217	bad_231b, 304a,
314a, 921b, 109, 117a, 122a, 128a, 131a, 205, 211a, 214 , 225b, bad_301a, 310a,
316b, 925b, 110, 119a, 124a, 128b, 131b, 207, 212 , 215a, 229b, bad_302a, 312b,
921a. Files were uploaded to /data06/larsoner/for_hank/brainstudio with
variants of:

$ rsync -a --rsh="ssh -o KexAlgorithms=diffie-hellman-group1-sha1" --partial --progress --include="*_raw.fif" --include="*_raw-1.fif" --exclude="*" /media/ktavabi/ALAYA/data/ilabs/badbaby/*/bad_114/raw_fif/* larsoner@kasga.ilabs.uw.edu:/data06/larsoner/for_hank/brainstudio
>>> mne.io.read_raw_fif('../mismatch/bad_114/raw_fif/bad_114_mmn_raw.fif', allow_maxshield='yes').info['meas_date'].strftime('%y%m%d')

Then repackaged manually into brainstudio/bad_baby/bad_*/*/ directories
based on the recording dates (or just using 111111 for simplicity).

Subjects who did not complete preprocessing:

- 223a: Preproc (Only 13/15 good ECG epochs found)

About half the time their HPI was no good, so throw them out.
"""  # noqa: E501

import os.path as op
import traceback
from pathlib import Path

import janitor  # noqa
import mnefun
import pandas as pd

from score import score

static = op.join(Path(__file__).parents[1], "static")

columns = [
    "subjid",
    "badch",
    "behavioral",
    "complete",
    "ses",
    "age",
    "gender",
    "headsize",
    "maternaledu",
    "paternaledu",
    "maternalethno",
    "paternalethno",
    "ecg",
]

meg_features = (
    pd.read_excel(op.join(static, "meg_covariates.xlsx"), sheet_name="mmn")
    .clean_names()
    .select_columns(columns)
    .encode_categorical(columns=columns)
    .rename_columns({"subjid": "id"})
    .filter_on("complete == 1", complement=False)
)

ecg_channel = {
    f"bad_{k}": v for k, v in zip(meg_features["id"], meg_features["ecg"])
}
bads = {
    f"bad_{k}": [v] for k, v in zip(meg_features["id"], meg_features["badch"])
}

good, bad = list(), list()
subjects = sorted(f"bad_{id_}" for id_ in meg_features["id"])
assert set(subjects) == set(ecg_channel)
assert len(subjects) == 68

params = mnefun.read_params(
    op.join(Path(__file__).parents[1], "processing", "badbaby.yml")
)
params.work_dir = "/media/ktavabi/ALAYA/data/ilabs/badbaby"
params.ecg_channel = ecg_channel
params.score = score

# Set what will run
good, bad = list(), list()
use_subjects = params.subjects
# use_subjects = ['bad_130a', 'bad_225b', 'bad_226b']

continue_on_error = False
for subject in use_subjects:
    params.subject_indices = [params.subjects.index(subject)]
    default = False
    try:
        mnefun.do_processing(
            params,
            fetch_raw=default,
            do_score=True,
            push_raw=default,
            do_sss=True,
            fetch_sss=default,
            do_ch_fix=True,
            gen_ssp=True,
            apply_ssp=True,
            write_epochs=True,
            gen_covs=True,
            gen_report=True,
            print_status=default,
        )
    except Exception:
        if not continue_on_error:
            raise
        traceback.print_exc()
        bad.append(subject)
    else:
        good.append(subject)
print(
    f"Successfully processed {len(good)}/{len(good) + len(bad)}, bad:\n{bad}")
