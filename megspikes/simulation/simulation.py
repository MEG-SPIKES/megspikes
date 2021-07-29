import os.path as op
import shutil
from pathlib import Path
from typing import Union

import mne
import numpy as np
from megspikes.casemanager.casemanager import CaseManager
from mne.datasets import sample
from scipy import signal
from scipy.misc import electrocardiogram

mne.set_log_level("ERROR")


class Simulation:

    def __init__(self, root_dir_path: Union[Path, str] = None,
                 case: Union[str, None] = None):
        if isinstance(root_dir_path, (str, Path)):
            if Path(root_dir_path).is_dir():
                self.root = Path(root_dir_path)
        else:
            self.root = Path(op.dirname(__file__))

        if isinstance(case, str):
            self.case = case
        else:
            self.case = 'sample'
        self.spikes_file = Path(op.dirname(__file__)) / 'data' / 'spikes.npy'
        self.case_info = Path(op.dirname(__file__)) / 'data' / 'case_info.xlsx'

    def simulate_dataset(self, length: int = 600,
                         spikes: Union[str, None] = None,
                         n_sources: int = 1):
        # length seconds
        # TODO: add spikes reading option
        if not isinstance(spikes, str):
            self.spike_shapes = np.load(self.spikes_file)
            # peaks in self.spike_shapes
            self.peak_times = [0.195, 0.110, 0.290, 0.260]

        self.load_mne_dataset()
        self._simulate_events(length, n_sources)
        self._simulate_raw()
        self._add_annotation()
        self.simulate_data_structure()
        self.simulate_case()

    def load_mne_dataset(self):
        data_path = sample.data_path()
        self.subjects_dir = op.join(data_path, 'subjects')
        self.mne_subject = 'sample'
        meg_path = op.join(data_path, 'MEG', self.mne_subject)

        # fif info
        fname_info = op.join(meg_path, 'sample_audvis_raw.fif')
        info = mne.io.read_info(fname_info, verbose=False)
        meg_channels = mne.pick_types(info, meg=True, exclude=[])
        self.info = mne.pick_info(info, meg_channels)
        self.info['bads'] = []
        self.info['sfreq'] = 1000  # Hz
        self.tstep = 1 / self.info['sfreq']

        # forward solution of the sample subject.
        fwd_fname = op.join(meg_path, 'sample_audvis-meg-oct-6-fwd.fif')
        self.fwd = mne.read_forward_solution(fwd_fname, verbose='error')
        self.fwd['info']['bads'] = []
        self.src = self.fwd['src']

        # raw
        raw_path = op.join(meg_path, 'sample_audvis_raw.fif')
        self.raw = mne.io.read_raw_fif(raw_path)

        self.meg_path = meg_path

    def _simulate_events(self, length=15, n_sources=1):
        self.n_events = length // n_sources
        self.n_sources = n_sources

        n_events = self.n_events
        n_sources = self.n_sources
        events = np.zeros((n_events * n_sources, 3))
        events[:, 0] = 1000 * np.arange(n_events * n_sources)
        events[:, 2] = sum([[n+1]*n_events for n in range(n_sources)], [])
        event_id = {
            'spike_shape_1': 1,
            'spike_shape_2': 2,
            'spike_shape_3': 3,
            'spike_shape_4': 4}

        # label, activation (nAm)
        activations = {
            'spike_shape_1': [('G_temp_sup-G_T_transv-rh', 120)],
        }
        # 'spike_shape_2': [('S_subparietal-rh', 280)],
        # 'spike_shape_3': [('S_subparietal-lh', 120)],
        # 'spike_shape_4': [('S_subparietal-lh', 120)],

        # SEE: https://europepmc.org/article/PMC/2937159
        annot = 'aparc.a2009s'

        # Load the necessary label names.
        label_names = sorted(
            set(activation[0] for activation_list in activations.values()
                for activation in activation_list))
        region_names = list(activations.keys())

        self.source_simulator = mne.simulation.SourceSimulator(
            self.src, tstep=self.tstep)

        for region_id, region_name in enumerate(region_names, 1):
            event_inx = region_id
            if region_id == 2:
                # To make events 1 and 2 simultaneous
                event_inx = 1
            events_tmp = events[np.where(events[:, 2] == event_inx)[0], :]
            # print(events_tmp.shape, event_inx, events_tmp[:, 0].max())
            i = 0
            label_name = activations[region_name][i][0]
            label_tmp = mne.read_labels_from_annot(
                self.mne_subject, annot, subjects_dir=self.subjects_dir,
                regexp=label_name, verbose=False)
            label_tmp = label_tmp[0]
            amplitude_tmp = activations[region_name][i][1]
            wf_tmp = 1e-10 * self.spike_shapes[region_id-1]
            # print(wf_tmp.shape)
            self.source_simulator.add_data(
                label_tmp, amplitude_tmp * wf_tmp, np.int32(events_tmp))

        self.label_names = label_names
        self.events = events
        self.annot = annot
        self.event_id = event_id
        self.activations = activations

    def _add_annotation(self):
        spikes = np.array(np.arange(self.n_events*self.n_sources),
                          dtype=np.float64)
        all_peaks = self.peak_times[:self.n_sources]
        spikes += np.array(
            [[i]*self.n_events for i in all_peaks]).flatten()
        spikes_annot = mne.Annotations(
            onset=spikes,  # in seconds
            duration=[0.001]*len(spikes),  # in seconds, too
            description=sum(
                [[f'SP{i+1}']*self.n_events for i in range(self.n_sources)],
                []))
        self.raw_simulation.set_annotations(spikes_annot)

    def _simulate_raw(self):
        iir_filter = mne.time_frequency.fit_iir_model_raw(
            self.raw, order=5, picks='meg', tmin=60, tmax=180)[1]

        rng = np.random.RandomState(0)
        self.raw_simulation = mne.simulation.simulate_raw(
            self.info, self.source_simulator, forward=self.fwd)
        snr = 20.
        noise_cov = mne.make_ad_hoc_cov(self.raw_simulation.info)
        # Scale the noise to achieve the desired SNR
        noise_cov['data'] *= (20. / snr) ** 2
        mne.simulation.add_noise(self.raw_simulation, cov=noise_cov,
                                 iir_filter=iir_filter, random_state=rng)

    def simulate_data_structure(self):
        shutil.copy(str(self.case_info), str(self.root))
        case_dir = self.root / self.case
        fraw = case_dir / 'MEG_data' / 'tsss_mc' / 'sample_raw_tsss_mc.fif'
        fraw2 = (case_dir / 'MEG_data' / 'tsss_mc_artefact_correction'
                 / 'sample_raw_tsss_mc_art_corr.fif')
        Path.mkdir(fraw.parent, exist_ok=True, parents=True)
        Path.mkdir(fraw2.parent, exist_ok=True, parents=True)
        ernoise = case_dir / 'MEG_data' / 'empty_room' / 'ernoise-cov.fif'
        Path.mkdir(ernoise.parent, exist_ok=True, parents=True)

        if hasattr(self, 'raw_simulation'):
            self.raw_simulation.save(str(fraw2), overwrite=True)
        else:
            data, _ = simulate_raw_fast(seconds=1, sampling_freq=1000)
            data.save(str(fraw2), overwrite=True)
            self.raw_simulation = data
        Path.mkdir(case_dir / 'forward_model', exist_ok=True, parents=True)

        shutil.copy(
            op.join(self.meg_path, 'sample_audvis_raw-trans.fif'),
            str(case_dir / 'forward_model' / 'checked_visually_trans.fif'))
        shutil.copy(op.join(self.meg_path, 'ernoise-cov.fif'), str(ernoise))

    def simulate_case(self):
        case = CaseManager(
            root=self.root, case='sample', free_surfer=self.subjects_dir)
        case.set_basic_folders()
        case.select_fif_file(case.run)
        case.prepare_forward_model()
        self.case_manager = case


def simulate_raw_fast(seconds: int = 2, sampling_freq: int = 200,
                      n_channels: int = 306):
    ch_names = [f'MEG{n:03}' for n in range(1, n_channels + 1)]
    ch_types = ['mag', 'grad', 'grad'] * 102
    info = mne.create_info(
        ch_names, ch_types=ch_types, sfreq=sampling_freq)

    esfreq = 360
    if sampling_freq < esfreq:
        data = electrocardiogram()[:seconds*esfreq]
        data = signal.resample(data, seconds*sampling_freq)
        # plt.plot(
        #     np.linspace(0, 2, 2*esfreq), data[:2*esfreq], 'go-',
        #     np.linspace(0, 2, 2*sfreq), data2[:2*sfreq], '.-')
    else:
        data = electrocardiogram()[:seconds*sampling_freq]

    raw_data = np.repeat(np.array([data]), n_channels, axis=0)
    noise = np.random.normal(0, .1, raw_data.shape)
    raw_data = 1e-9 * (raw_data + noise)
    raw = mne.io.RawArray(raw_data, info)
    del raw_data
    return raw, data
