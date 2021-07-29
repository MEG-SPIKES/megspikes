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


@pytest.fixture(name='fname')
def fixture_data():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_detection'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.fixture(name="db")
def make_database():
    n_ica_comp = 4
    db = Database(meg_data_length=10_000, n_ica_components=n_ica_comp)
    return db


@pytest.fixture(name="dataset")
def make_dataset(db, fname):
    n_ica_comp = 4
    n_channels = 204
    raw_fif, cardio_ts = simulate_raw_fast(10, 1000)
    raw_fif.save(fname=fname / 'raw_test.fif', overwrite=True)

    ds = db.make_empty_dataset()
    for sens in [0, 1]:
        sel = dict(run=0, sensors=sens)
        name = "ica_sources"
        ica_sources = np.array([cardio_ts]*n_ica_comp)*5
        ds[sel][name][:, :] = ica_sources
        name = "ica_components"
        ds[sel][name][:, :] = np.random.sample((n_ica_comp, n_channels))
        name = "ica_components_localization"
        ds[sel][name][:, :] = np.random.sample((n_ica_comp, 3))
        name = "ica_components_gof"
        ds[sel][name][:] = np.array([72., 85., 94., 99.])
        name = "ica_components_kurtosis"
        ds[sel][name][:] = np.array([2, 0.5, 8, 0])
    return ds


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_ica_decomposition(db, dataset, fname, sensors):
    n_ica_comp = 4
    dataset[dict(run=0, sensors=0)].ica_components.values *= 0
    dataset[dict(run=0, sensors=1)].ica_components.values *= 0

    pd = PrepareData(data_file=fname / 'raw_test.fif', sensors=sensors)
    ds_channles = db.select_sensors(dataset, sensors, 0)
    decomposition = DecompositionICA(n_components=n_ica_comp)
    (ds_channles, data) = pd.fit_transform(ds_channles)
    _ = decomposition.fit_transform((ds_channles, data))
    assert dataset.ica_sources.loc[sensors, :, :].any()

    assert dataset.ica_components.loc[sensors, 0, 50].values != 0
    if sensors == 'mag':
        assert dataset.ica_components.loc[sensors, 0, 200].values == 0


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_components_selection(db, dataset, sensors):
    ds_channles = db.select_sensors(dataset, sensors, 0)
    selection = ComponentsSelection()
    (results, _) = selection.fit_transform((ds_channles, None))
    assert sum(results["ica_components_selected"].values) == 2

    ds_channles = db.select_sensors(dataset, sensors, 1)
    selection = ComponentsSelection(run=1)
    (results, _) = selection.fit_transform((ds_channles, None))


@pytest.mark.parametrize("run,n_runs", [
    (0, 1), (0, 2), (1, 2), (2, 3),
    (0, 4), (1, 4), (2, 4), (3, 4)])
@pytest.mark.parametrize("n_components", [4, 10, 20])
def test_components_selection_detailed(run, n_runs, n_components):
    selection = ComponentsSelection(run=run, n_runs=n_runs)
    components = np.random.sample((n_components, 102))
    kurtosis = np.random.uniform(low=0, high=30, size=(n_components,))
    gof = np.random.uniform(low=0, high=100, size=(n_components,))
    sel = selection.select_ica_components(components, kurtosis, gof)
    assert sum(sel) > 0


@pytest.mark.happy
def test_peaks_detection(db, dataset):
    name = "ica_components_selected"
    dataset[name][:] = np.array([1., 1., 1., 1.])

    ds = db.select_sensors(dataset, 'grad', 0)
    peak_detection = PeakDetection(prominence=2., width=1.)
    (results, _) = peak_detection.fit_transform((ds, None))
    assert results["ica_peaks_timestamps"].values.any()


def test_peaks_detection_details():
    pass
