import os.path as op
from pathlib import Path

import pytest

from megspikes.simulation.simulation import Simulation
from megspikes.casemanager.casemanager import CaseManager

sample_path = Path(op.dirname(__file__)).parent.parent.parent
sample_path = sample_path / 'example'
sample_path.mkdir(exist_ok=True)

sim = Simulation(sample_path)
sim.load_mne_dataset()
sim.simulate_data_structure()


@pytest.mark.happy
@pytest.mark.slow
def test_new_case():
    case = CaseManager(root=sample_path, case='sample',
                       free_surfer=sim.subjects_dir)
    assert case.case == 'sample'
    case.set_basic_folders()
    case.select_fif_file(case.run)
    case.prepare_forward_model()
    assert len(case.fwd) == 2
