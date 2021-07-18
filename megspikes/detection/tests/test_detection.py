import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.database.database import Database
from megspikes.detection.detection import (DecompositionICA,
                                           ComponentsSelection)
from megspikes.utils import PrepareData
from megspikes.simulation.simulation import simulate_raw_fast

# from scipy.signal import find_peaks

sample_path = Path(op.dirname(__file__)).parent.parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)
raw_fif, _ = simulate_raw_fast(10, 1000)
fname = sample_path / 'raw_test.fif'
raw_fif.save(fname=fname, overwrite=True)
n_ica_comp = 4
db = Database(n_ica_components=n_ica_comp)


@pytest.fixture(name="sensors")
def fixture_data():
    return ['grad', 'mag']


@pytest.fixture(name="dataset")
def make_dataset():
    ds = db.make_empty_dataset()
    for sens in [0, 1]:
        sel = dict(run=0, sensors=sens)
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
        (ds_channles, data) = pd.transform(ds_channles)
        decomposition.fit((ds_channles, data))
        _ = decomposition.transform((ds_channles, data))


def test_components_selection(sensors, dataset):
    for sens in sensors:
        ds_channles = db.select_sensors(dataset, sens, 0)
        selection = ComponentsSelection()
        (results, _) = selection.transform((ds_channles, raw_fif))
        assert sum(results["ica_components_selected"].values) == 2

        ds_channles = db.select_sensors(dataset, sens, 1)
        selection = ComponentsSelection(run=1)
        (results, _) = selection.transform((ds_channles, raw_fif))
