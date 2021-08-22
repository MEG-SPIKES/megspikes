import os

import numpy as np
import yaml
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


aspire_alphacsc_params = os.path.join(
    os.path.dirname(__file__), "aspire_alphacsc_default_params.yml")


class MakePipeline():
    def __init__(self, case: CaseManager) -> None:
        self.case = case


def aspire_alphacsc_pipeline(case: CaseManager, update_params: dict):
    with open(aspire_alphacsc_params, 'rt') as f:
        default_params = yaml.safe_load(f.read())
    params = update_default_params(default_params, update_params)

    pipe_sensors = []
    pipe_runs = []

    for sens in ['grad', 'mag']:
        for run in params['runs']:
            pipe_runs.append(
                (f"run_{run}",
                 Pipeline([
                    ('prepare_data',
                     PrepareData(data_file=case.fif_file, sensors=sens,
                                 **params['PrepareData'])),
                    ('load_select_dataset',
                     LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(**params['PeakDetection'])),
                    ('peaks_localization',
                     PeakLocalization(case=case, sensors=sens,
                                      **params['PeakLocalization'])),
                    ('peaks_cleaning',
                     CleanDetections(**params['CleanDetections'])),
                    ('alphacsc_decomposition',
                     DecompositionAlphaCSC(**params['DecompositionAlphaCSC'])),
                    ('alphacsc_components_localization',
                     AlphaCSCComponentsLocalization(
                         case=case, sensors=sens,
                         **params['AlphaCSCComponentsLocalization'])),
                    ('alphacsc_events_selection',
                     SelectAlphacscEvents(
                         sensors=sens, **params['SelectAlphacscEvents'])),
                    ('save_dataset',
                     SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('finish_one_run', ToFinish())
                    ])))
        pipe_sensors.append(
            (sens,
             Pipeline([
                ('prepare_data',
                 PrepareData(data_file=case.fif_file, sensors=sens,
                             **params['PrepareData'])),
                ('load_dataset',
                 LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('ica_decomposition',
                 DecompositionICA(**params['DecompositionICA'])),
                ('components_localization',
                 ICAComponentsLocalization(
                     case=case, sensors=sens,
                     **params['ICAComponentsLocalization'])),
                ('save_dataset',
                 SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('finish_one_sensors_before_run', ToFinish()),
                ('spikes_detection', FeatureUnion(pipe_runs)),
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('load_data',
         PrepareData(data_file=case.fif_file, sensors=True,
                     **params['PrepareData'])),
        ('prepare_aspire_alphacsc_dataset',
         PrepareAspireAlphacscDataset(
             fif_file=case.fif_file, fwd=case.fwd['ico5'],
             **params['PrepareAspireAlphacscDataset'])),
        ('save_empty_dataset', SaveDataset(dataset=case.dataset)),
        ('finish_preparation', ToFinish()),
        ('extract_all_atoms',  FeatureUnion(pipe_sensors)),  # no output
        ('prepare_data',
         PrepareData(data_file=case.fif_file, sensors=True,
                     **params['PrepareData'])),
        ('load_aspire_alphacsc_dataset',
         LoadDataset(dataset=case.dataset, sensors=None, run=None)),
        ('merge_atoms', AspireAlphacscRunsMerging(
            **params['AspireAlphacscRunsMerging'])),
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


def update_default_params(defaults: dict, updates: dict):
    def recursive_update(params, key, val):
        for k, v in params.items():
            if isinstance(v, dict):
                recursive_update(v, key, val)
            elif key == k:
                params[k] = val
        return params

    for key, val in updates.items():
        assert key in defaults.keys(), (
            f"{key} is not in the defaults params file")
        if isinstance(val, dict):
            update_default_params(defaults[key], val)
        else:
            defaults = recursive_update(defaults, key, val)
    return defaults
