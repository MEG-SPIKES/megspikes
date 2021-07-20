from megspikes.casemanager.casemanager import CaseManager
from megspikes.database.database import LoadDataset, SaveDataset
from megspikes.detection.detection import (ComponentsSelection,
                                           DecompositionICA, PeakDetection,
                                           CleanDetections)
from megspikes.localization.localization import (ComponentsLocalization,
                                                 PeakLocalization)
from megspikes.utils import PrepareData, ToTest
from sklearn.pipeline import Pipeline, FeatureUnion


def make_full_pipeline(case: CaseManager, n_ica_components: int = 20,
                       n_ica_peaks: int = 2000, n_cleaned_peaks: int = 300):
    pipe_sensors = []
    pipe_runs = []

    for sens in ['grad', 'mag']:
        for run in range(4):
            pipe_runs.append(
                (f'run_{run}',
                 Pipeline([
                    ('prepare_data', PrepareData(sensors=sens, resample=200.)),
                    ('load_select_dataset',
                     LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('select_ica_components', ComponentsSelection(run=run)),
                    ('detect_ica_peaks',
                     PeakDetection(prominence=2., width=1.,
                                   n_detections_threshold=n_ica_peaks)),
                    ('peaks_localization',
                     PeakLocalization(case=case, sensors=sens)),
                    ('peaks_cleaning',
                     CleanDetections(n_cleaned_peaks=n_cleaned_peaks)),
                    ('save_dataset',
                     SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                    ('test', ToTest())
                    ])))
        pipe_sensors.append(
            (sens,
             Pipeline([
                ('prepare_data',
                 PrepareData(data_file=case.fif_file, sensors=sens)),
                ('select_dataset',
                 LoadDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('ica_decomposition',
                 DecompositionICA(n_components=n_ica_components)),
                ('components_localization',
                 ComponentsLocalization(case=case, sensors=sens)),
                ('save_dataset',
                    SaveDataset(dataset=case.dataset, sensors=sens, run=run)),
                ('spikes_detection', FeatureUnion(pipe_runs))
                ])))
        pipe_runs = []
    pipe = Pipeline([
        ('make_clusters_library',  FeatureUnion(pipe_sensors))])
    return pipe
