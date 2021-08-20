import numpy as np
import pytest
from megspikes.scoring.scoring import distance_to_resection_hull


@pytest.mark.parametrize(
    "resection,detection,output",
    [(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      np.array([[0, 2, 0]]), 1),
     (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      np.array([[0, 2, 0], [2, 0, 0]]), 1),
     (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      np.array([[0, 0.5, 0], [1.5, 0, 0]]), 0),
     (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
      np.array([[0, 0.5, 0]]), 0),   # round
     (np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]),
      np.array([[0, 5, 0]]), -5),  # on the border
     (np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]),
      np.array([[3, 3, 0]]), -4),  # one point inside the hull
     (np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]]),
      np.array([[3, 3, 0], [3, 3, 1]]), -4)])  # two point inside the hull
def test_distance_to_resection(resection, detection, output):
    dist = distance_to_resection_hull(resection, detection)
    assert dist == output
