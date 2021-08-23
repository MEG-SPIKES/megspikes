import os.path as op
from pathlib import Path

import pytest

from megspikes.simulation.simulation import Simulation
from megspikes.casemanager.casemanager import CaseManager


@pytest.fixture(name='simulation')
def run_simulation():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_casemanager'
    sample_path.mkdir(exist_ok=True, parents=True)
    sim = Simulation(sample_path)
    sim.simulate_dataset()
    return sim


@pytest.mark.happy
@pytest.mark.slow
def test_new_case(simulation):
    case = CaseManager(root=simulation.root, case='sample',
                       free_surfer=simulation.subjects_dir)
    assert case.case == 'sample'
    case.set_basic_folders()
    case.select_fif_file(case.run)
    case.prepare_forward_model()
    assert len(case.fwd) == 2
