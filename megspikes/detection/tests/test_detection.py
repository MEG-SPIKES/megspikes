import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.database.database import Database
from megspikes.detection.detection import (DecompositionICA,
                                           ComponentsSelection,
                                           PeakDetection)
from megspikes.utils import PrepareData
from megspikes.simulation.simulation import simulate_raw_fast

sample_path = Path(op.dirname(__file__)).parent.parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)
raw_fif, cardio_ts = simulate_raw_fast(10, 1000)
fname = sample_path / 'raw_test.fif'
raw_fif.save(fname=fname, overwrite=True)
n_ica_comp = 4
db = Database(meg_data_length=10_000,
              n_ica_components=n_ica_comp)


@pytest.fixture(name="sensors")
def fixture_data():
    return ['grad', 'mag']


@pytest.fixture(name="dataset")
def make_dataset():
    ds = db.make_empty_dataset()
    for sens in [0, 1]:
        sel = dict(run=0, sensors=sens)
        name = "ica_sources"
        ica_sources = np.array([cardio_ts]*n_ica_comp)*5
        ds[sel][name][:, :] = ica_sources
        name = "ica_components_localization"
        ds[sel][name] = (
            ("ica_component", "mni_coordinates"),
            np.random.sample((n_ica_comp, 3)))
        name = "ica_components_gof"
        ds[sel][name][:] = np.array([72., 85., 94., 99.])
        name = "ica_components_kurtosis"
        ds[sel][name][:] = np.array([2, 0.5, 8, 0])
    return ds


@pytest.mark.happy
def test_ica_decomposition(sensors, dataset):
    for sens in sensors:
        pd = PrepareData(fname, sens)
        ds_channles = db.select_sensors(dataset, sens, 0)
        decomposition = DecompositionICA(n_components=n_ica_comp)
        (ds_channles, data) = pd.fit_transform(ds_channles)
        _ = decomposition.fit_transform((ds_channles, data))


def test_components_selection(sensors, dataset):
    for sens in sensors:
        ds_channles = db.select_sensors(dataset, sens, 0)
        selection = ComponentsSelection()
        (results, _) = selection.fit_transform((ds_channles, raw_fif))
        assert sum(results["ica_components_selected"].values) == 2

        ds_channles = db.select_sensors(dataset, sens, 1)
        selection = ComponentsSelection(run=1)
        (results, _) = selection.fit_transform((ds_channles, raw_fif))


def test_peaks_detection(dataset):
    name = "ica_components_selected"
    dataset[name][:] = np.array([1., 1., 1., 1.])

    ds = db.select_sensors(dataset, 'grad', 0)
    peak_detection = PeakDetection(prominence=2., width=1.)
    (results, _) = peak_detection.fit_transform((ds, raw_fif))
    assert results["ica_peaks_timestamps"].values.any()
    del results, peak_detection

    # test no peaks
    ds = db.select_sensors(dataset, 'grad', 0)
    peak_detection = PeakDetection(prominence=100., width=10000.)
    (results, _) = peak_detection.fit_transform((ds, raw_fif))
    assert not results["ica_peaks_timestamps"].all()
