from typing import Any, Tuple

import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull, Delaunay
from sklearn.base import BaseEstimator, TransformerMixin

from ..database.database import check_and_read_from_dataset


def distance_to_resection_hull(resection_mni_points: np.ndarray,
                               detection_mni_points: np.ndarray):
    """Estimate distance between detections the resection's border source.
       If the detection is inside the resection area then the function returns
       negative value.

    Parameters
    ----------
    resection_mni_points : np.array, (n_sources, 3)
        resection sources in MNI space
    detection_mni_points : np.array, (n_sources, 3)
        detection sources in MNI space

    Returns
    -------
    int
        distance in mm; negative means inside the hull
    """
    assert len(resection_mni_points.shape) == 2, "Wrong input shape"
    assert len(detection_mni_points.shape) == 2, "Wrong input shape"
    chull = ConvexHull(resection_mni_points)
    dhull = Delaunay(resection_mni_points)
    hvertices = resection_mni_points[chull.vertices, :]
    distance = 0
    for det in detection_mni_points:
        nearest_vert = np.linalg.norm(hvertices - det, ord=2, axis=1).min()
        if dhull.find_simplex(det) >= 0:
            # inside the hull
            distance += -nearest_vert
        else:
            # outside the hull
            distance += nearest_vert
    distance /= detection_mni_points.shape[0]
    return round(distance, 0)


class ScoreIZPrediction(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X: xr.Dataset, Any, y=None):
        return self

    def transform(self, X) -> Tuple[xr.Dataset, Any]:
        return X

    def score(self, X: xr.Dataset, y: np.ndarray, slope_point: str = "peak"):
        self.detection_stc = check_and_read_from_dataset(
            X, 'iz_prediction',
            dict(iz_prediction_timepoint=slope_point))
        self.fwd_mni = check_and_read_from_dataset(
            X, 'fwd_mni_coordinates')
        self.detection_mni = self.fwd_mni[self.detection_stc > 0]
        return distance_to_resection_hull(y, self.detection_mni)
