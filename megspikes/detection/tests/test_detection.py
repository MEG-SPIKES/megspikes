# -*- coding: utf-8 -*-
from megspikes.detection.detection import (
    DecompositionICA)
from megspikes.database.database import Database
from megspikes.utils import prepare_data, simulate_raw
from sklearn.pipeline import make_pipeline
import numpy as np
from scipy.signal import find_peaks
import pytest

pytest
raw = ''
cardio_ts = 0


@pytest.fixture(name="fif_data")
def fixture_data():
    raw_fif, _ = simulate_raw(10, 1000)
    fname = 'raw_test.fif'
    raw_fif.save(fname=fname, overwrite=True)
    return [prepare_data(fname, 'grad'),
            prepare_data(fname, 'mag')]


@pytest.fixture(name="dataset")
def fixture_dataset():
    db = Database(meg_data_length=10_000)
    ds = db.make_empty_dataset()
    ds_grad = ds.sel(
        sensors='grad', decomposition_sensors_type='grad', run=0).squeeze()
    ds_mag = ds.sel(
        sensors='mag', decomposition_sensors_type='mag', run=0).squeeze()
    return [ds_grad, ds_mag]


@pytest.mark.happy
def test_ica_decomposition(fif_data, dataset):
    for data, ds in zip(fif_data, dataset):
        decomposition = DecompositionICA(n_components=20)
        decomposition.fit(ds, data)
        _ = decomposition.transform(ds)


@pytest.mark.pipeline
@pytest.mark.happy
def test_pipeline(fif_data, dataset):
    for data, ds in zip(fif_data, dataset):
        pipe = make_pipeline(DecompositionICA(n_components=20))
        _ = pipe.fit_transform(ds, data)
