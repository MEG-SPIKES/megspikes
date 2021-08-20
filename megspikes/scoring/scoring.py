import numpy as np
from scipy.spatial import ConvexHull, Delaunay


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
