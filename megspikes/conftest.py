
import pytest
import numpy as np
from megspikes.simulation.simulation import Simulation


@pytest.fixture(scope="module", name="dataset")
def make_full_dataset(db):
    ds = db.make_empty_dataset()
    for sens in db.sensors:
        selection_ch = ds.attrs[sens]
        selection_sens = dict(sensors=sens)
        shape = ds["ica_sources"].loc[selection_sens].shape
        ds["ica_sources"].loc[selection_sens] = np.random.sample(shape)

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
            ds['ica_component_selection'].loc[
                selection_pipe_sens] = np.random.sample(shape)

            shape = ds['detection_properties'].loc[
                selection_pipe_sens].shape
            ds['detection_properties'].loc[
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


@pytest.fixture(scope="module", name='simulation')
def run_simulation(test_sample_path):
    test_sample_path.mkdir(exist_ok=True, parents=True)

    sim = Simulation(test_sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=5)
    return sim
