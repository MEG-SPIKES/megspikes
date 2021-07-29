import os.path as op
from pathlib import Path

import pytest

from megspikes.simulation.simulation import Simulation


@pytest.mark.happy
def test_simulation():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'example'
    sample_path.mkdir(exist_ok=True)

    sim = Simulation(sample_path)
    sim.load_mne_dataset()
    sim.simulate_dataset(length=5)
