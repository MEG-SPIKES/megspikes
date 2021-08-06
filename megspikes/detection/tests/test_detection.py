import os.path as op
from pathlib import Path

import numpy as np
import pytest
from megspikes.database.database import Database, select_sensors
from megspikes.detection.detection import (DecompositionICA,
                                           ComponentsSelection,
                                           PeakDetection,
                                           CleanDetections,
                                           DecompositionAlphaCSC,
                                           SelectAlphacscEvents)
from megspikes.utils import PrepareData


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path2():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_detection'
    return sample_path


@pytest.fixture(name='spikes')
def spikes_waveforms():
    root = Path(op.dirname(__file__)).parent.parent
    spikes = root / 'simulation' / 'data' / 'spikes.npy'
    return np.load(spikes)


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_ica_decomposition(db, dataset, simulation, sensors):
    pipeline = "aspire_alphacsc_run_1"
    n_ica_comp = db.n_ica_components
    _, sel = select_sensors(dataset, 'grad', pipeline)
    dataset.loc[sel].ica_components.values *= 0
    _, sel = select_sensors(dataset, 'mag', pipeline)
    dataset.loc[sel].ica_components.values *= 0

    pd = PrepareData(data_file=simulation.case_manager.fif_file,
                     sensors=sensors, resample=dataset.time.attrs['sfreq'])
    ds_channles = db.select_sensors(dataset, sensors, pipeline)
    ds_channles.ica_component_properties.loc[:, 'kurtosis'] *= 0

    decomposition = DecompositionICA(n_components=n_ica_comp)
    (ds_channles, data) = pd.fit_transform(ds_channles)
    (ds_channles, data) = decomposition.fit_transform((ds_channles, data))
    assert ds_channles.ica_sources.loc[:, :].any()
    assert ds_channles.ica_components.loc[:, :].any()
    assert ds_channles.ica_component_properties.loc[:, 'kurtosis'].any()


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
@pytest.mark.parametrize(
    "gof,kurtosis,sel_comp,run",
    [([72., 85., 94., 99.], [2, 0.5, 8, 0], [1, 0, 1, 0], 0),
     ([72., 85., 94., 99.], [2, 0.5, 8, 0], [1, 0, 0, 0], 1)])  # FIXME:
def test_components_selection(dataset, sensors, gof, kurtosis, sel_comp, run):
    dataset["ica_component_properties"].loc[
        dict(sensors=sensors, ica_component_property="gof")
        ] = np.array(gof)
    dataset["ica_component_properties"].loc[
        dict(sensors=sensors, ica_component_property="kurtosis")
        ] = np.array(kurtosis)
    pipeline = f"aspire_alphacsc_run_{run+1}"
    ds_channles, _ = select_sensors(dataset, sensors, pipeline)
    ds_channles['ica_component_selection'] *= 0
    selection = ComponentsSelection(run=run)
    (results, _) = selection.fit_transform((ds_channles, None))
    assert sum(results['ica_component_selection'].values) == sum(sel_comp)


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
    dataset['ica_component_selection'] *= 0
    dataset['ica_component_selection'] += 1
    ds_grad = db.select_sensors(dataset, 'grad', "aspire_alphacsc_run_1")
    peak_detection = PeakDetection(prominence=2., width=1.)
    (results, _) = peak_detection.fit_transform((ds_grad, None))
    # assert results["ica_peaks_timestamps"].values.any()


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
    (detected_peaks, channles) = peak_detection.find_ica_peaks(
        data, np.array([1, 1, 1, 1]))
    detected_peaks -= sfreq
    for i in true_peaks:
        assert min(np.abs(detected_peaks - i)) < 20
    # with raises(ValueError, match="1-D array"):
    #     find_peaks(np.array(1))


@pytest.mark.parametrize(
    'times,subcorr,selection,n_cleaned_peaks,diff_threshold',
    [([400, 500, 700, 900, 1200], [0.1, 0.9, 0.9, 0.9, 0.1],
      [0, 1, 1, 1, 0], 3, 0.5),
     ([400, 500, 700, 900, 1200], [0.1, 0.9, 0.9, 0.9, 0.2],
      [0, 1, 1, 1, 1], 4, 0.5),
     ([400, 500], [0.1, 0.9],
      [0, 1], 10, 4)])
def test_detection_cleaning_details(times, subcorr, selection,
                                    n_cleaned_peaks, diff_threshold):
    clean = CleanDetections(diff_threshold=diff_threshold,
                            n_cleaned_peaks=n_cleaned_peaks)
    result = clean.clean_detections(np.array(times), np.array(subcorr), 200.)
    assert (result == selection).all()


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_alphacsc_decomposition(simulation, db, dataset, sensors):
    pipeline = "aspire_alphacsc_run_1"

    pd = PrepareData(data_file=simulation.case_manager.fif_file,
                     sensors=sensors, resample=200.)
    ds_channles = db.select_sensors(dataset, sensors, pipeline)

    ds_channles.detection_properties.loc[
        dict(detection_property='selected_for_alphacsc')] *= 0
    ds_channles.detection_properties.loc[
        dict(detection_property='selected_for_alphacsc')][
            [500, 600, 700]] = 1
    (ds_channles, data) = pd.fit_transform(ds_channles)
    alpha = DecompositionAlphaCSC()
    results, _ = alpha.fit_transform((ds_channles, data))
    # TODO: add tests


@pytest.mark.happy
@pytest.mark.parametrize("sensors", ["grad", "mag"])
def test_alphacsc_events_selection(dataset, db, simulation, sensors):
    pipeline = "aspire_alphacsc_run_1"
    pd = PrepareData(data_file=simulation.case_manager.fif_file,
                     sensors=sensors, resample=200.)
    ds_channles = db.select_sensors(dataset, sensors, pipeline)
    ds_channles.detection_properties.loc[
        dict(detection_property='selected_for_alphacsc')] *= 0
    ica_peaks = [300, 350, 500, 600, 700]
    ds_channles.detection_properties.loc[
        dict(detection_property='selected_for_alphacsc')][
            ica_peaks] = 1
    z_peaks = [i - 60 for i in ica_peaks]
    ds_channles.alphacsc_z_hat[:, z_peaks] = 20
    sel = SelectAlphacscEvents(sensors=sensors)
    (ds_channles, data) = pd.fit_transform(ds_channles)
    sel.fit_transform((ds_channles, data))


@pytest.mark.parametrize(
    "z_hat,ica_peaks,n_detections",
    [([0]*100 + [1] + [0]*99, [0]*199 + [1], 0),
     ([0]*120 + [1] + [0]*79, [0]*199 + [1], 1)])
def test_alphacsc_events_selection_details(z_hat, ica_peaks, n_detections):
    alpha_select = SelectAlphacscEvents()
    detection, _ = alpha_select._find_max_z(
        np.array(z_hat), np.array(ica_peaks), 1)
    assert sum(detection) == n_detections
