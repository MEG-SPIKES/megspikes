import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.database.database import Database, select_sensors
from megspikes.detection.detection import (DecompositionICA,
                                           ComponentsSelection,
                                           PeakDetection)
from megspikes.utils import PrepareData
from megspikes.simulation.simulation import simulate_raw_fast


@pytest.fixture(name='fname')
def sample_path():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_detection'
    sample_path.mkdir(exist_ok=True, parents=True)
    return sample_path


@pytest.fixture(name='spikes')
def spikes_waveforms():
    root = Path(op.dirname(__file__)).parent.parent
    spikes = root / 'simulation' / 'data' / 'spikes.npy'
    return np.load(spikes)


@pytest.fixture(scope="module", name="db")
def make_database():
    n_ica_comp = 4
    db = Database(times=np.linspace(0, 10, 10_000),
                  n_ica_components=n_ica_comp)
    return db


@pytest.fixture(name="ds")
def make_dataset(db, fname, dataset):
    ds = dataset.copy(deep=True)
    n_ica_comp = 4
    raw_fif, cardio_ts = simulate_raw_fast(10, 1000)
    raw_fif.save(fname=fname / 'raw_test.fif', overwrite=True)
    for sens in db.sensors:
        ds["ica_sources"].loc[
            dict(sensors=sens)
            ] = np.array([cardio_ts]*n_ica_comp)*5
        ds["ica_component_properties"].loc[
            dict(sensors=sens, ica_component_property="gof")
            ] = np.array([72., 85., 94., 99.])
        ds["ica_component_properties"].loc[
            dict(sensors=sens, ica_component_property="kurtosis")
            ] = np.array([2, 0.5, 8, 0])
    return ds


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_ica_decomposition(db, ds, fname, sensors):
    pipeline = "aspire_alphacsc_run_1"
    n_ica_comp = 4
    _, sel = select_sensors(ds, 'grad', pipeline)
    ds.loc[sel].ica_components.values *= 0
    _, sel = select_sensors(ds, 'mag', pipeline)
    ds.loc[sel].ica_components.values *= 0

    pd = PrepareData(data_file=fname / 'raw_test.fif', sensors=sensors)
    ds_channles = db.select_sensors(ds, sensors, pipeline)
    ds_channles.ica_component_properties.loc[:, 'kurtosis'] *= 0

    decomposition = DecompositionICA(n_components=n_ica_comp)
    (ds_channles, data) = pd.fit_transform(ds_channles)
    (ds_channles, data) = decomposition.fit_transform((ds_channles, data))
    assert ds_channles.ica_sources.loc[:, :].any()
    assert ds_channles.ica_components.loc[:, :].any()
    assert ds_channles.ica_component_properties.loc[:, 'kurtosis'].any()


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_components_selection(ds, sensors):
    pipeline = "aspire_alphacsc_run_1"
    ds_channles, _ = select_sensors(ds, sensors, pipeline)
    ds_channles['ica_component_selection'] *= 0
    selection = ComponentsSelection()
    (results, _) = selection.fit_transform((ds_channles, None))
    assert sum(results['ica_component_selection'].values) == 2

    pipeline = "aspire_alphacsc_run_2"
    ds_channles, _ = select_sensors(ds, sensors, pipeline)
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


@pytest.mark.parametrize('prominence', [1, 2, 4, 7.])
@pytest.mark.parametrize('width', [1., 5, 10.])
def test_peaks_detection_details(spikes, prominence, width):
    sfreq = 10_000
    peak_detection = PeakDetection(
        prominence=prominence, width=width, sfreq=sfreq)
    true_peaks = []
    for spike in spikes:
        data = np.zeros(sfreq*3)
        data[sfreq:sfreq+500] = spike
        (detected_peaks, _) = peak_detection._find_peaks(data)
        detected_peaks -= sfreq
        true_peaks.append(np.argmax(spike))
        assert len(detected_peaks) > 0
        assert min(np.abs(detected_peaks - np.argmax(spike))) < 20

    data = np.zeros((4, sfreq*3))
    data[:, sfreq:sfreq+500] = spikes
    detected_peaks = peak_detection.find_ica_peaks(
        data, np.array([1, 1, 1, 1]))
    detected_peaks -= sfreq
    for i in true_peaks:
        assert min(np.abs(detected_peaks - i)) < 20
    # with raises(ValueError, match="1-D array"):
    #     find_peaks(np.array(1))
