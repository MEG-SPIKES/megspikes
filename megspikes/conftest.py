
import pytest
import numpy as np
from megspikes.simulation.simulation import Simulation, simulate_raw_fast
from megspikes.database.database import Database


@pytest.fixture(scope="module", name='simulation')
def run_simulation(test_sample_path):
    test_sample_path.mkdir(exist_ok=True, parents=True)

    sim = Simulation(test_sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=10)
    return sim


@pytest.fixture(scope="module", name="db")
def make_database(simulation):
    n_ica_comp = 4
    n_atoms = 3
    case = simulation.case_manager
    db = Database(n_atoms=n_atoms, n_ica_components=n_ica_comp)
    db.read_case_info(case.fif_file, case.fwd['ico5'], sfreq=200.)
    return db


@pytest.fixture(scope="module", name="dataset")
def make_full_dataset(db):
    ds = db.make_empty_dataset()
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

        for pipe in db.pipelines_names:
            selection_pipe_sens = dict(
                pipeline=pipe, sensors=sens)
            selection_pipe_ch = dict(
                pipeline=pipe, channel=selection_ch)

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
    return ds
