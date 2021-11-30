import numpy as np
import pytest

from megspikes.database.database import read_meg_info_for_database
from megspikes.simulation.simulation import Simulation, simulate_raw_fast
import mne

mne.set_log_level("ERROR")


@pytest.fixture(scope="module", name='simulation')
def run_simulation(test_sample_path):
    test_sample_path.mkdir(exist_ok=True, parents=True)
    sim = Simulation(test_sample_path, n_events=[10, 0, 0, 0])
    sim.simulate_dataset()
    return sim


@pytest.fixture(scope="module", name='simulation_large')
def run_large_simulation(test_sample_path):
    test_sample_path.mkdir(exist_ok=True, parents=True)
    sim = Simulation(test_sample_path, n_events=[5, 5, 5, 5])
    sim.simulate_dataset(noise_scaler=2)
    return sim


@pytest.fixture(scope="module", name='mne_example_dataset')
def run_mne_example_simulation(test_sample_path):
    path = test_sample_path / 'mne_example_dataset'
    path.mkdir(exist_ok=True, parents=True)
    sim = Simulation(path, n_events=[10, 0, 0, 0])
    sim.simulate_dataset_mne_example()
    return sim


@pytest.fixture(scope="module", name='simulation_epochs_grad')
def simulation_epochs_grad(simulation):
    raw = simulation.raw_simulation.copy()
    raw.filter(2, 90)
    events = mne.events_from_annotations(raw)
    epochs_grad = mne.Epochs(
        raw, events[0], events[1], tmin=-0.5, tmax=0.5,
        baseline=None, preload=True, reject_by_annotation=False,
        proj=False, picks='grad')
    return epochs_grad


@pytest.fixture(scope="module", name="aspire_alphacsc_random_dataset")
def prepare_aspire_alphacsc_random_dataset(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    sfreq = 200.
    raw = simulation.raw_simulation.copy().resample(sfreq, npad="auto")
    n_ica_comp = 4
    n_atoms = 2
    atom_length = 0.5  # seconds
    n_runs = 4

    ds = db.make_aspire_alphacsc_dataset(
        times=raw.times,
        n_ica_components=n_ica_comp,
        n_atoms=n_atoms,
        atom_length=atom_length,
        n_runs=n_runs,
        sfreq=sfreq)

    for sens in db.sensors:
        selection_ch = ds.attrs[sens]
        selection_sens = dict(sensors=sens)
        shape = ds["ica_sources"].loc[selection_sens].shape
        # set ECG as ica sources timeseries
        _, cardio_ts = simulate_raw_fast(
            15, ds.time.attrs["sfreq"])
        ds["ica_sources"].loc[selection_sens] = np.array([
            cardio_ts[:shape[1]]]*shape[0])*5

        shape = ds["ica_components"].loc[:, selection_ch].shape
        ds["ica_components"].loc[:, selection_ch] = np.random.sample(shape)

        shape = ds['ica_component_properties'].loc[selection_sens].shape
        ds['ica_component_properties'].loc[
            selection_sens] = np.random.sample(shape)

        for run in ds.run.values:
            selection_pipe_sens = dict(
                run=run, sensors=sens)
            selection_pipe_ch = dict(
                run=run, channel=selection_ch)

            shape = ds['ica_component_selection'].loc[
                selection_pipe_sens].shape
            data = np.random.sample(shape)
            mean = data.mean()
            data[data >= mean] = 1
            data[data < mean] = 0
            ds['ica_component_selection'].loc[
                selection_pipe_sens] = data

            shape = ds['detection_properties'].loc[
                selection_pipe_sens].shape
            ds['detection_properties'].loc[
                selection_pipe_sens] = np.random.sample(shape)
            det = np.random.choice(a=[0, 1], size=shape[-1], p=[0.9, 0.1])
            ds.detection_properties.loc[
                selection_pipe_sens].loc[dict(
                detection_property='alphacsc_detection')] = det
            ds.detection_properties.loc[
                selection_pipe_sens].loc[dict(
                detection_property='alphacsc_atom')] = np.random.randint(
                0, n_atoms, shape[-1])

            shape = ds['alphacsc_atoms_properties'].loc[
                selection_pipe_sens].shape
            ds['alphacsc_atoms_properties'].loc[
                selection_pipe_sens] = np.random.sample(shape)
            ds['alphacsc_atoms_properties'].loc[
                selection_pipe_sens].loc[dict(
                alphacsc_atom_property='selected')] = np.random.choice(
                a=[0, 1], size=n_atoms, p=[0.5, 0.5])

            shape = ds['alphacsc_z_hat'].loc[selection_pipe_sens].shape
            ds['alphacsc_z_hat'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['alphacsc_v_hat'].loc[selection_pipe_sens].shape
            ds['alphacsc_v_hat'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['alphacsc_u_hat'].loc[selection_pipe_ch].shape
            ds['alphacsc_u_hat'].loc[
                selection_pipe_ch] = np.random.sample(shape)
    selection_spikes = dict(atoms_library_property='library_detection')
    ds.alphacsc_atoms_library_properties.loc[selection_spikes][
        np.int32(simulation.detections*0.2)] = 1
    selection_clust = dict(atoms_library_property='library_cluster')
    ds.alphacsc_atoms_library_properties.loc[selection_clust][
        np.int32(simulation.detections*0.2)] = simulation.clusters - 1
    ds.to_netcdf(simulation.case_manager.dataset)
    return ds


@pytest.fixture(scope="module", name="aspire_alphacsc_empty_dataset")
def prepare_aspire_alphacsc_empty_dataset(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    sfreq = 200.
    raw = simulation.raw_simulation.copy().resample(sfreq, npad="auto")
    n_ica_comp = 4
    n_atoms = 2
    atom_length = 0.5  # seconds
    n_runs = 4

    ds = db.make_aspire_alphacsc_dataset(
        times=raw.times,
        n_ica_components=n_ica_comp,
        n_atoms=n_atoms,
        atom_length=atom_length,
        n_runs=n_runs,
        sfreq=sfreq)
    return ds


@pytest.fixture(scope="module", name="clusters_empty_dataset")
def prepare_clusters_empty_dataset(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    raw = simulation.raw_simulation
    n_clusters = 5
    ds = db.make_clusters_dataset(
        times=raw.times,
        n_clusters=n_clusters,
        evoked_length=1.,  # second
        sfreq=1000.)
    return ds


@pytest.fixture(scope="module", name="clusters_random_dataset")
def prepare_clusters_random_dataset(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    raw = simulation.raw_simulation
    n_clusters = 2
    ds = db.make_clusters_dataset(
        times=raw.times,
        n_clusters=n_clusters,
        evoked_length=1.,  # second
        sfreq=1000.)
    true_spike_peaks = np.int32(
        np.unique(simulation.spikes) * raw.info['sfreq'])
    clusters = np.zeros_like(true_spike_peaks)
    clusters[4:] = 1
    ds.spike.loc[dict(detection_property='detection')][true_spike_peaks] = 1
    ds.spike.loc[dict(detection_property='cluster')][
        true_spike_peaks] = clusters
    ds.spike.loc[dict(detection_property='sensor')][
        true_spike_peaks] = clusters

    ds.cluster_properties.loc[dict(cluster_property='cluster_id')] = [0, 1]
    ds.cluster_properties.loc[dict(cluster_property='sensors')] = [0, 1]
    return ds
