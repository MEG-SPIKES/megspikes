from typing import List

import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline

from .casemanager.casemanager import CaseManager
from .database.database import (LoadDataset, PrepareAspireAlphacscDataset,
                                PrepareClustersDataset, SaveDataset,
                                read_meg_info_for_database)
from .detection.detection import (AspireAlphacscRunsMerging, CleanDetections,
                                  ComponentsSelection, DecompositionAlphaCSC,
                                  DecompositionICA, ManualDetections,
                                  PeakDetection, SelectAlphacscEvents)
from .localization.localization import (AlphaCSCComponentsLocalization,
                                        ClustersLocalization, ForwardToMNI,
                                        ICAComponentsLocalization,
                                        PeakLocalization, PredictIZClusters)
from .utils import PrepareData, ToFinish


class MakePipeline():
    def __init__(self, case: CaseManager) -> None:
        self.case = case


def aspire_alphacsc_pipeline(case: CaseManager,
                             n_ica_components: int = 20,
                             resample: float = 200.,
                             n_ica_peaks: int = 2000,
                             n_cleaned_peaks: int = 300,
                             n_atoms: int = 3,
                             atoms_width: float = 0.5,
                             z_hat_threshold: float = 3.,
                             z_hat_threshold_min: float = 1.5,
                             runs: List[int] = [0, 1, 2, 3]):
    pipe_sensors = []
    pipe_runs = []
    runs = [int(i) for i in runs]

    for sens in ['grad', 'mag']:
        for run in runs:
            pipe_runs.append(
                (f"run_{run}",
                 Pipeline([
                    ('prepare_data',
                     PrepareData(data_file=case.fif_file, sensors=sens,
                                 resample=resample)),
                    ('load_select_dataset',
                     LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(prominence=8., width=2.,
                                   n_detections_threshold=n_ica_peaks)),
                    ('peaks_localization',
                     PeakLocalization(case=case, sensors=sens)),
                    ('peaks_cleaning',
                     CleanDetections(n_cleaned_peaks=n_cleaned_peaks)),
                    ('alphacsc_decomposition',
                     DecompositionAlphaCSC(n_atoms=n_atoms, sfreq=resample,
                                           atoms_width=atoms_width)),
                    ('alphacsc_components_localization',
                     AlphaCSCComponentsLocalization(case=case, sensors=sens)),
                    ('alphacsc_events_selection',
                     SelectAlphacscEvents(
                         sensors=sens, n_atoms=n_atoms,
                         z_hat_threshold=z_hat_threshold,
                         z_hat_threshold_min=z_hat_threshold_min,
                         sfreq=resample)),
                    ('save_dataset',
                     SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('finish_one_run', ToFinish())
                    ])))
        pipe_sensors.append(
            (sens,
             Pipeline([
                ('prepare_data',
                 PrepareData(data_file=case.fif_file, sensors=sens,
                             resample=resample)),
                ('prepare_aspire_alphacsc_dataset',
                 PrepareAspireAlphacscDataset(
                     fif_file=case.fif_file, fwd=case.fwd['ico5'],
                     atoms_width=atoms_width, n_runs=len(runs),
                     n_ica_comp=n_ica_components, n_atoms=n_atoms)),
                ('save_empty_dataset',
                 SaveDataset(dataset=case.dataset)),
                ('load_dataset',
                 LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('ica_decomposition',
                 DecompositionICA(n_components=n_ica_components)),
                ('components_localization',
                 ICAComponentsLocalization(case=case, sensors=sens)),
                ('save_dataset',
                 SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('finish_one_sensors_before_run', ToFinish()),
                ('spikes_detection', FeatureUnion(pipe_runs)),
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('extract_all_atoms',  FeatureUnion(pipe_sensors)),  # no output
        ('prepare_data',
         PrepareData(data_file=case.fif_file, sensors=True,
                     resample=resample)),
        ('load_aspire_alphacsc_dataset',
         LoadDataset(dataset=case.dataset, sensors=None, run=None)),
        ('merge_atoms', AspireAlphacscRunsMerging(runs=runs, n_atoms=n_atoms)),
        ('save_dataset', SaveDataset(dataset=case.dataset))])
    return pipe


def iz_prediction_pipeline(case: CaseManager, detection_sfreq: float = 200.):
    pipe = Pipeline([
        ('prepare_clusters_dataset',
         PrepareClustersDataset(fif_file=case.fif_file, fwd=case.fwd['ico5'],
                                detection_sfreq=detection_sfreq)),
        ('localize_clusters', ClustersLocalization(case=case)),
        ('convert_forward_to_mni', ForwardToMNI(case=case)),
        ('predict_IZ', PredictIZClusters(case=case)),
        ('save_dataset', SaveDataset(dataset=case.cluster_dataset))
        ])
    return pipe


def manual_pipeline(case: CaseManager, detections: np.ndarray,
                    clusters: np.ndarray):
    database = read_meg_info_for_database(case.fif_file, case.fwd['ico5'])

    pipe = Pipeline([
        ('make_clusters_library', ManualDetections(
            case.cluster_dataset, database, detections, clusters)),
        ('localize_clusters', ClustersLocalization(case=case)),
        ('convert_forward_to_mni', ForwardToMNI(case=case)),
        ('predict_IZ', PredictIZClusters(case=case)),
        ('save_dataset', SaveDataset(dataset=case.manual_cluster_dataset))
    ])
    return pipe
