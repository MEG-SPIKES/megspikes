import os.path as op
from pathlib import Path

import pytest

import numpy as np

import mne
from megspikes.casemanager.casemanager import CaseManager
from megspikes.localization.localization import (ComponentsLocalization)
from megspikes.simulation.simulation import Simulation

import xarray as xr

sample_path = Path(op.dirname(__file__)).parent.parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)

sim = Simulation(sample_path)
sim.load_mne_dataset()
sim.simulate_dataset(length=1)

case = CaseManager(root=sample_path, case='sample',
                   free_surfer=sim.subjects_dir)
case.set_basic_folders()
case.select_fif_file(case.run)
case.prepare_forward_model()


@pytest.fixture(name="dataset")
def make_dataset():
    n_ica_comp = 3
    ica_components = xr.DataArray(
        np.ones((n_ica_comp, 306)),
        dims=("ica_component", "channels"))
    ica_components_localization = xr.DataArray(
        np.random.sample((n_ica_comp, 3)),
        dims=("ica_component", "mni_coord"))
    ica_components_gof = xr.DataArray(
        np.random.sample(n_ica_comp),
        dims=("ica_component"))
    ds = xr.Dataset(data_vars={
        "ica_components": ica_components,
        "ica_components_localization": ica_components_localization,
        "ica_components_gof": ica_components_gof})
    return ds


def test_components_localization(dataset):
    cl = ComponentsLocalization(case=case)
    cl.fit((dataset, sim.raw_simulation))
    results = cl.fit_transform((dataset, sim.raw_simulation))
    assert results[0]['ica_components_localization'].any()
    assert results[0]['ica_components_gof'].any()
