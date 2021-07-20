import os.path as op
from pathlib import Path

import pytest
from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import Database
from megspikes.simulation.simulation import Simulation
from megspikes.pipeline import make_full_pipeline

sample_path = Path(op.dirname(__file__)).parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)

sim = Simulation(sample_path)
sim.load_mne_dataset()
sim.simulate_dataset(length=10)


@pytest.mark.pipeline
@pytest.mark.happy
@pytest.mark.slow
def test_pipeline():
    case = CaseManager(root=sample_path, case='sample',
                       free_surfer=sim.subjects_dir)
    case.set_basic_folders()
    case.select_fif_file(case.run)
    case.prepare_forward_model()
    n_ica_components = 3
    db = Database(n_ica_components=n_ica_components)
    db.read_case_info(case.fif_file, case.fwd['ico5'])
    ds = db.make_empty_dataset()
    ds.to_netcdf(case.dataset)

    pipe = make_full_pipeline(case, n_ica_components=n_ica_components)
    _ = pipe.fit_transform(None)
