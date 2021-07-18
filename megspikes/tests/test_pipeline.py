import pytest
from megspikes.database.database import Database
from megspikes.detection.detection import DecompositionICA
from megspikes.utils import PrepareData, simulate_raw
from sklearn.pipeline import make_pipeline


@pytest.fixture(name="file_sensors")
def fixture_data():
    raw_fif, _ = simulate_raw(10, 1000)
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


@pytest.mark.pipeline
@pytest.mark.happy
def test_pipeline(file_sensors, dataset):
    for (file, sensors), ds in zip(file_sensors, dataset):
        pipe = make_pipeline(
            PrepareData(file, sensors),
            DecompositionICA(n_components=20))
        _ = pipe.fit_transform(ds)
