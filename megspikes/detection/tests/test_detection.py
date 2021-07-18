# import numpy as np
import pytest
from megspikes.database.database import Database
from megspikes.detection.detection import DecompositionICA
from megspikes.utils import PrepareData
from megspikes.simulation.simulation import simulate_raw_fast
# from scipy.signal import find_peaks


@pytest.fixture(name="file_sensors")
def fixture_data():
    raw_fif, _ = simulate_raw_fast(10, 1000)
    fname = 'raw_test.fif'
    raw_fif.save(fname=fname, overwrite=True)
    return [[fname, 'grad'],
            [fname, 'mag']]


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
def test_ica_decomposition(file_sensors, dataset):
    for (file, sensors), ds in zip(file_sensors, dataset):
        pd = PrepareData(file, sensors)
        decomposition = DecompositionICA(n_components=20)
        (ds, data) = pd.transform(ds)
        decomposition.fit((ds, data))
        _ = decomposition.transform(ds)
