import os.path as op
from pathlib import Path

import pytest


@pytest.fixture(scope="module", name="test_sample_path")
def sample_path2():
    sample_path = Path(op.dirname(__file__)).parent.parent.parent
    sample_path = sample_path / 'tests_data' / 'test_visualization'
    return sample_path


def test_cluster_slope_viewer(simulation):
    pass
