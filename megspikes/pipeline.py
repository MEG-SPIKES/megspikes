import os
from typing import Dict, Tuple

import yaml
from sklearn.pipeline import FeatureUnion, Pipeline

from .casemanager.casemanager import CaseManager
from .database.database import (LoadDataset, PrepareAspireAlphacscDataset,
                                PrepareClustersDataset, ReadDetectionResults,
                                SaveDataset)
from .detection.detection import (AspireAlphacscRunsMerging, CleanDetections,
                                  ComponentsSelection, DecompositionAlphaCSC,
                                  DecompositionICA, PeakDetection,
                                  SelectAlphacscEvents)
from .localization.localization import (AlphaCSCComponentsLocalization,
                                        ClustersLocalization, ForwardToMNI,
                                        ICAComponentsLocalization,
                                        PeakLocalization, PredictIZClusters)
from .utils import PrepareData, ToFinish

aspire_alphacsc_params = os.path.join(
    os.path.dirname(__file__), "aspire_alphacsc_default_params.yml")
clusters_params = os.path.join(
    os.path.dirname(__file__), "clusters_default_params.yml")


def aspire_alphacsc_pipeline(case: CaseManager, update_params: dict,
                             rewrite_previous_results: bool = False,
                             manual_ica_components: Dict[str, Tuple[
                                 Tuple[int]]] = None):
    """Create ASPIRE AlphaCSC pipeline object.

    Parameters
    ----------
    case : CaseManager
        This object includes head model and the link to the MEG recording
    update_params : dict
        Parameters to update
    rewrite_previous_results : bool
        Rewrite previous results
    manual_ica_components : Dict[str: Tuple[Tuple[int]]], by default {
            'grad': None, 'mag': None}
        Manually ICA components per sensor type

    Returns
    -------
    sklearn.pipeline.Pipeline
        [description]
    """
    if manual_ica_components is None:
        manual_ica_components = {
            'grad': None, 'mag': None}
    if (not rewrite_previous_results) & case.dataset.is_file():
        raise RuntimeError(
            'Results dataset exists and you try to overwrite it. If you want to'
            'do that, set rewrite_previous_results=True')

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
                    ('select_ica_components',
                     ComponentsSelection(run=run,
                                         manual_ica_components_selection=
                                         manual_ica_components[sens],
                                         **params['ComponentsSelection'])),
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
        ('save_empty_dataset', SaveDataset(
            dataset=case.dataset,
            rewrite_previous_results=rewrite_previous_results)),
        ('finish_preparation', ToFinish()),
        ('extract_all_atoms',  FeatureUnion(pipe_sensors)),  # no output
        ('prepare_data',
         PrepareData(data_file=case.fif_file, sensors=True,
                     **params['PrepareData'])),
        ('load_aspire_alphacsc_dataset',
         LoadDataset(dataset=case.dataset, sensors=None, run=None)),
        ('merge_atoms', AspireAlphacscRunsMerging(
            **params['AspireAlphacscRunsMerging'])),
        ('save_dataset', SaveDataset(
            dataset=case.dataset,
            rewrite_previous_results=rewrite_previous_results))])
    return pipe


def iz_prediction_pipeline(case: CaseManager, update_params: dict,
                           rewrite_previous_results: bool = False):

    if (not rewrite_previous_results) & case.cluster_dataset.is_file():
        raise RuntimeError(
            'Results dataset exists and you try to overwrite it. If you want to'
            'do that, set rewrite_previous_results=True')

    with open(clusters_params, 'rt') as f:
        default_params = yaml.safe_load(f.read())
    params = update_default_params(default_params, update_params)
    pipe = Pipeline([
        ('prepare_clusters_dataset',
         PrepareClustersDataset(fif_file=case.fif_file, fwd=case.fwd['ico5'],
                                **params['PrepareClustersDataset'])),
        ('localize_clusters',
         ClustersLocalization(case=case, **params['ClustersLocalization'])),
        ('convert_forward_to_mni', ForwardToMNI(case=case)),
        ('predict_IZ',
         PredictIZClusters(case=case, **params['PredictIZClusters'])),
        ('save_dataset', SaveDataset(
            dataset=case.cluster_dataset,
            rewrite_previous_results=rewrite_previous_results))
        ])
    return pipe


def read_detection_iz_prediction_pipeline(
        case: CaseManager, clusters_params: dict,
        rewrite_previous_results: bool = False):
    if (not rewrite_previous_results) & case.cluster_dataset.is_file():
        raise RuntimeError(
            'Results dataset exists and you try to overwrite it. If you want to'
            'do that, set rewrite_previous_results=True')

    pipe = Pipeline([
        ('read_detections_results', ReadDetectionResults()),
        ('iz_prediction_pipeline',
         iz_prediction_pipeline(
             case, clusters_params, rewrite_previous_results))])
    return pipe


def update_default_params(defaults: dict, updates: dict):
    """Update default pipeline parameters.

    Parameters
    ----------
    defaults : dict
        default pipeline parameters
    updates : dict
        parameters to update
    """
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
