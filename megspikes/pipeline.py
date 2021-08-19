from typing import List

from sklearn.pipeline import FeatureUnion, Pipeline

from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import Database, LoadDataset, SaveDataset
from megspikes.detection.detection import (CleanDetections,
                                           ComponentsSelection,
                                           DecompositionAlphaCSC,
                                           DecompositionICA, PeakDetection,
                                           SelectAlphacscEvents,
                                           AspireAlphacscRunsMerging)
from megspikes.localization.localization import (
    AlphaCSCComponentsLocalization, ClustersLocalization,
    ICAComponentsLocalization, PeakLocalization, PredictIZClusters)
from megspikes.utils import PrepareData, ToFinish


def aspire_alphacsc_pipeline(case: CaseManager,
                             database: Database,
                             n_ica_components: int = 20,
                             resample: float = 200.,
                             n_ica_peaks: int = 2000,
                             n_cleaned_peaks: int = 300,
                             n_atoms: int = 3,
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
                     PrepareData(sensors=sens, resample=resample)),
                    ('load_select_dataset',
                     LoadDataset(
                         dataset=case.dataset, sensors=sens,
                         run=run)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(prominence=8., width=2.,
                                   n_detections_threshold=n_ica_peaks)),
                    ('peaks_localization',
                     PeakLocalization(case=case, sensors=sens)),
                    ('peaks_cleaning',
                     CleanDetections(n_cleaned_peaks=n_cleaned_peaks)),
                    ('alphacsc_decomposition',
                     DecompositionAlphaCSC(n_atoms=n_atoms, sfreq=resample)),
                    ('alphacsc_components_localization',
                     AlphaCSCComponentsLocalization(case=case, sensors=sens)),
                    ('alphacsc_events_selection',
                     SelectAlphacscEvents(
                         sensors=sens, n_atoms=n_atoms,
                         z_hat_threshold=z_hat_threshold,
                         z_hat_threshold_min=z_hat_threshold_min,
                         sfreq=resample)),
                    ('save_dataset',
                     SaveDataset(
                         dataset=case.dataset, sensors=sens, run=run)),
                    ('test', ToFinish())
                    ])))
        pipe_sensors.append(
            (sens,
             Pipeline([
                ('prepare_data',
                 PrepareData(
                     data_file=case.fif_file,
                     sensors=sens,
                     resample=resample)),
                ('select_dataset',
                 LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('ica_decomposition',
                 DecompositionICA(n_components=n_ica_components)),
                ('components_localization',
                 ICAComponentsLocalization(case=case, sensors=sens)),
                ('save_dataset',
                    SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('spikes_detection', FeatureUnion(pipe_runs))
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('extract_all_atoms',  FeatureUnion(pipe_sensors)),
        ('prepare_data', PrepareData(data_file=case.fif_file, sensors=True)),
        ('make_clusters_library', AspireAlphacscRunsMerging(
            detection_dataset=case.dataset,
            clusters_dataset=case.cluster_dataset,
            database=database,
            runs=runs, n_atoms=n_atoms)),
        ('localize_clusters', ClustersLocalization(case=case)),
        # ('predict_IZ', PredictIZClusters(case=case)),
        ('save_dataset', SaveDataset(dataset=case.cluster_dataset))
        ])
    return pipe

