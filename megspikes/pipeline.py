from sklearn.pipeline import FeatureUnion, Pipeline

from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import (LoadDataset, SaveDataset,
                                         SaveFullDataset)
from megspikes.detection.detection import (CleanDetections, ClustersMerging,
                                           ComponentsSelection,
                                           CropDataAroundPeaks,
                                           DecompositionAlphaCSC,
                                           DecompositionICA, PeakDetection,
                                           SelectAlphacscEvents)
from megspikes.localization.localization import (
    AlphaCSCComponentsLocalization, ClustersLocalization,
    ICAComponentsLocalization, PeakLocalization, PredictIZClusters)
from megspikes.utils import PrepareData, ToFinish


def aspike_alphacsc_pipeline(case: CaseManager, n_ica_components: int = 20,
                             resample: float = 200., n_ica_peaks: int = 2000,
                             n_cleaned_peaks: int = 300, n_atoms=3,
                             z_hat_threshold=3., z_hat_threshold_min=1.5,
                             pipe_names=['aspire_alphacsc_run_1']):
    pipe_sensors = []
    pipe_runs = []

    for sens in ['grad', 'mag']:
        for run, pipe_name in enumerate(pipe_names):
            pipe_runs.append(
                (pipe_name,
                 Pipeline([
                    ('prepare_data',
                     PrepareData(sensors=sens, resample=resample)),
                    ('load_select_dataset',
                     LoadDataset(
                         dataset=case.dataset, sensors=sens,
                         pipeline=pipe_name)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(prominence=8., width=2.,
                                   n_detections_threshold=n_ica_peaks)),
                    # ('peaks_localization',
                    #  PeakLocalization(case=case, sensors=sens)),
                    # ('peaks_cleaning',
                    #  CleanDetections(n_cleaned_peaks=n_cleaned_peaks)),
                    # ('crop_data', CropDataAroundPeaks()),
                    # ('alphacsc_decomposition',
                    #  DecompositionAlphaCSC(n_atoms=n_atoms, sfreq=resample)),
                    # ('alphacsc_components_localization',
                    #  AlphaCSCComponentsLocalization(case=case, sensors=sens)),
                    # ('alphacsc_events_selection',
                    #  SelectAlphacscEvents(
                    #      sensors=sens, n_atoms=n_atoms,
                    #      z_hat_threshold=z_hat_threshold,
                    #      z_hat_threshold_min=z_hat_threshold_min,
                    #      sfreq=resample)),
                    ('save_dataset',
                     SaveDataset(
                         dataset=case.dataset, sensors=sens,
                         pipeline=pipe_name)),
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
                 LoadDataset(dataset=case.dataset, sensors=sens,
                             pipeline=pipe_name)),
                ('ica_decomposition',
                 DecompositionICA(n_components=n_ica_components)),
                ('components_localization',
                 ICAComponentsLocalization(case=case, sensors=sens)),
                ('save_dataset',
                    SaveDataset(dataset=case.dataset, sensors=sens,
                                pipeline=pipe_name)),
                ('spikes_detection', FeatureUnion(pipe_runs))
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('extract_all_atoms',  FeatureUnion(pipe_sensors)),
        # ('make_clusters_library', ClustersMerging(dataset=case.dataset)),
        # ('prepare_data', PrepareData(data_file=case.fif_file, sensors=True)),
        # ('localize_clusters', ClustersLocalization(
        #     case=case,
        #     db_name_detections='clusters_library_timestamps',
        #     db_name_clusters='clusters_library_cluster_id',
        #     detection_sfreq=resample)),
        # ('predict_IZ', PredictIZClusters(case=case)),
        # ('save_dataset', SaveFullDataset(dataset=case.dataset))
        ])
    return pipe
