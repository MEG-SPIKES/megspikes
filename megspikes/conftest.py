import pytest
import numpy as np
from megspikes.simulation.simulation import Simulation, simulate_raw_fast
from megspikes.database.database import read_meg_info_for_database


@pytest.fixture(scope="module", name='simulation')
def run_simulation(test_sample_path):
    test_sample_path.mkdir(exist_ok=True, parents=True)
    sim = Simulation(test_sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=10)
    return sim


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

            shape = ds['alphacsc_atoms_properties'].loc[
                selection_pipe_sens].shape
            ds['alphacsc_atoms_properties'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['alphacsc_z_hat'].loc[selection_pipe_sens].shape
            ds['alphacsc_z_hat'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['alphacsc_v_hat'].loc[selection_pipe_sens].shape
            ds['alphacsc_v_hat'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['alphacsc_u_hat'].loc[selection_pipe_ch].shape
            ds['alphacsc_u_hat'].loc[
                selection_pipe_ch] = np.random.sample(shape)
    ds.to_netcdf(simulation.case_manager.dataset)
    return ds


@pytest.fixture(scope="module", name="aspire_alphacsc_empty_dataset")
def prepare_aspire_alphacsc_empty_dataset(simulation):
    case = simulation.case_manager
    db = read_meg_info_for_database(
        simulation.case_manager.fif_file, case.fwd['ico5'])
    sfreq = 200.
    raw = simulation.raw_simulation.resample(sfreq, npad="auto")
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
