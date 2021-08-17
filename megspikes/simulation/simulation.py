import os.path as op
import shutil
from pathlib import Path
from typing import List, Union, Tuple

import mne
import numpy as np
import nibabel as nb
from mne.datasets import sample
from mne.source_space import _check_mri
from scipy import signal
from scipy.misc import electrocardiogram

from ..casemanager.casemanager import CaseManager
from ..utils import stc_to_nifti

mne.set_log_level("ERROR")


class Simulation:
    def __init__(self, root_dir_path: Union[Path, str] = None,
                 time_between_events: int = 1, atlas: str = 'aparc.a2009s'):
        if isinstance(root_dir_path, (str, Path)):
            if Path(root_dir_path).is_dir():
                self.root = Path(root_dir_path)
        else:
            self.root = Path(op.dirname(__file__))
        self.time_between_events = time_between_events
        # SEE: https://europepmc.org/article/PMC/2937159
        self.atlas = atlas

        # path to the data reqired for the simulation
        self.spikes_file = Path(op.dirname(__file__)) / 'data' / 'spikes.npy'
        assert self.spikes_file.is_file(), (
            "No spikes.npy in the simulation data folder:"
            f" {self.spikes_file.parent}")
        self.case_info = Path(op.dirname(__file__)) / 'data' / 'case_info.xlsx'
        assert self.case_info.is_file(), (
            "No case_info.xlsx in the simulation data folder:"
            f" {self.case_info.parent}")

        # Load spikes
        self.spike_shapes = np.load(self.spikes_file)

        # Events id
        self.event_id = {
            'spike_shape_1': 1,
            'spike_shape_2': 2,
            'spike_shape_3': 3,
            'spike_shape_4': 4}

        # label, activation (nAm)
        self.activations = {
            'spike_shape_1': [('G_temp_sup-G_T_transv-rh', 120)],
            'spike_shape_2': [('S_subparietal-rh', 280)],
            'spike_shape_3': [('S_subparietal-lh', 120)],
            'spike_shape_4': [('S_subparietal-lh', 120)]
        }

        # Set approximate peak times for each spike in self.spike_shapes
        self.peak_times = [0.195, 0.110, 0.290, 0.260]

    def simulate_dataset(self, n_events: List[int] = [15, 0, 0, 0],
                         simultaneous: List[bool] = [False]*4,
                         sfreq: float = 1000., snr: float = 1.):
        # length seconds
        # TODO: add spikes reading option
        info, fwd, raw = self._read_mne_sample_dataset()
        events = self._prepare_events(n_events, simultaneous, sfreq=sfreq)
        source_simulator = self._simulate_sources(events, fwd['src'], 1/sfreq)
        simulation = self._simulate_raw(source_simulator, fwd, info, raw, snr)
        simulation, spikes = self._add_annotation(simulation, events, sfreq)
        self.spikes = spikes  # seconds
        self._simulate_data_structure(simulation)
        self._simulate_case()
        self.events = events
        self.raw_simulation = simulation

    def _read_mne_sample_dataset(self) -> Tuple[mne.Info, mne.Forward,
                                                mne.io.Raw]:
        """Load sample MNE dataset.

        Notes
        -----
        MNE-Python dataset description:
        https://mne.tools/stable/overview/datasets_index.html#sample
        """
        data_path = sample.data_path()
        self.subjects_dir = op.join(data_path, 'subjects')
        self.mne_subject = 'sample'
        meg_path = op.join(data_path, 'MEG', self.mne_subject)
        self.meg_path = meg_path

        # fif info
        fname_info = op.join(meg_path, 'sample_audvis_raw.fif')
        info = mne.io.read_info(fname_info)
        meg_channels = mne.pick_types(info, meg=True, exclude=[])
        info = mne.pick_info(info, meg_channels)
        info['bads'] = []
        info['sfreq'] = 1000  # Hz
        # self.tstep = 1 / info['sfreq']

        # forward solution
        fwd_fname = op.join(meg_path, 'sample_audvis-meg-oct-6-fwd.fif')
        fwd = mne.read_forward_solution(fwd_fname)
        fwd['info']['bads'] = []
        # self.src = self.fwd['src']

        # fif raw
        raw_path = op.join(meg_path, 'sample_audvis_raw.fif')
        raw = mne.io.read_raw_fif(raw_path)
        return info, fwd, raw

    def _prepare_events(self, n_events: List[int] = [15, 0, 0, 0],
                        simultaneous: List[bool] = [False]*4,
                        sfreq: float = 1000) -> np.ndarray:
        all_sources_events = []
        first_time = 0
        t_steps = self.time_between_events * sfreq
        for s, n in enumerate(n_events):
            tmp_events = np.zeros((n_events[s], 3))
            last_time = first_time + n
            # events timepoints
            tmp_events[:, 0] = t_steps * np.arange(first_time, last_time)
            # events id
            tmp_events[:, 2] = np.repeat(s+1, n)
            all_sources_events.append(tmp_events)

            if not simultaneous[s]:
                first_time += n

        events = np.vstack(all_sources_events)
        return events

    def _simulate_sources(self, events: np.ndarray, src: mne.SourceSpaces,
                          tstep: float, scaler: float = 1e-10
                          ) -> mne.simulation.SourceSimulator:
        activations = self.activations
        # Load the necessary label names.
        label_names = sorted(
            set(activation[0] for activation_list in activations.values()
                for activation in activation_list))
        region_names = list(activations.keys())
        self.labels = []

        source_simulator = mne.simulation.SourceSimulator(
            src, tstep=tstep)

        for region_name in region_names:
            event_inx = self.event_id[region_name]
            events_tmp = events[np.where(events[:, 2] == event_inx)[0], :]
            label_name = activations[region_name][0][0]
            label_tmp = mne.read_labels_from_annot(
                self.mne_subject, self.atlas, subjects_dir=self.subjects_dir,
                regexp=label_name)[0]
            self.labels.append(label_tmp)
            amplitude_tmp = activations[region_name][0][1]
            wf_tmp = scaler * self.spike_shapes[event_inx - 1]
            source_simulator.add_data(
                label_tmp, amplitude_tmp * wf_tmp, np.int32(events_tmp))
        self.label_names = label_names
        return source_simulator

    def _simulate_raw(self, source_simulator: mne.simulation.SourceSimulator,
                      fwd: mne.Forward, info: mne.Info, raw: mne.io.Raw,
                      snr: float = 1.) -> mne.io.Raw:
        iir_filter = mne.time_frequency.fit_iir_model_raw(
            raw, order=5, picks='meg', tmin=60, tmax=180)[1]
        rng = np.random.RandomState(0)
        simulation = mne.simulation.simulate_raw(
            info, source_simulator, forward=fwd)
        noise_cov = mne.make_ad_hoc_cov(simulation.info)
        # Scale the noise to achieve the desired SNR
        noise_cov['data'] *= snr
        mne.simulation.add_noise(simulation, cov=noise_cov,
                                 iir_filter=iir_filter, random_state=rng)
        return simulation

    def _add_annotation(self, simulation: mne.io.Raw, events: np.ndarray,
                        sfreq: float = 1000.):
        times = events[:, 0] / sfreq
        sources = np.unique(events[:, 2])
        labels = [f"SRC{int(i)}" for i in events[:, 2]]
        for s in sources:
            times[events[:, 2] == s] += self.peak_times[int(s)-1]

        spikes_annot = mne.Annotations(
            onset=times,  # in seconds
            duration=[0.001]*len(times),  # in seconds
            description=labels)
        return simulation.set_annotations(spikes_annot), times

    def _simulate_data_structure(self, simulation: mne.io.Raw):
        shutil.copy(str(self.case_info), str(self.root))

        # Create folder
        case_dir = self.root / self.mne_subject
        fraw = case_dir / 'MEG_data' / 'tsss_mc' / 'sample_raw_tsss_mc.fif'
        fraw2 = (case_dir / 'MEG_data' / 'tsss_mc_artefact_correction'
                 / 'sample_raw_tsss_mc_art_corr.fif')
        Path.mkdir(fraw.parent, exist_ok=True, parents=True)
        Path.mkdir(fraw2.parent, exist_ok=True, parents=True)
        ernoise = case_dir / 'MEG_data' / 'empty_room' / 'ernoise-cov.fif'
        Path.mkdir(ernoise.parent, exist_ok=True, parents=True)
        Path.mkdir(case_dir / 'forward_model', exist_ok=True, parents=True)

        # Save simulation
        simulation.save(str(fraw2), overwrite=True)

        # Copy .trans file
        shutil.copy(
            op.join(self.meg_path, 'sample_audvis_raw-trans.fif'),
            str(case_dir / 'forward_model' / 'checked_visually_trans.fif'))
        shutil.copy(op.join(self.meg_path, 'ernoise-cov.fif'), str(ernoise))

    def _simulate_case(self):
        case = CaseManager(
            root=self.root, case=self.mne_subject,
            free_surfer=self.subjects_dir)
        case.set_basic_folders()
        case.select_fif_file(case.run)
        case.prepare_forward_model()
        self.case_manager = case
        resection_path = case.basic_folders['resection mask']
        self.fresection = resection_path.with_suffix('.nii')

        # copy and save T1 image
        mri_fname = _check_mri('T1.mgz', self.mne_subject, self.subjects_dir)
        t1 = nb.load(mri_fname)
        nb.save(t1, self.fresection.with_name("T1.nii"))

        # Save resection
        # self.save_resection_as_nifti(self.fresection)

    def save_resection_as_nifti(self, fsave: Union[str, Path]):
        # TODO: add more labels
        label = self.labels[0]
        vertices = [i['vertno'] for i in self.fwd['src']]
        hemi = 0 if label.hemi == 'lh' else 1
        data = []
        for i in [0, 1]:
            data.append(np.zeros(len(vertices[i])))
            if i == hemi:
                lab_data = [
                    1 if i in label.vertices else 0 for i in vertices[hemi]]
                data[i] = np.array(lab_data)
        label_mni = mne.vertex_to_mni(
            vertices[hemi][data[hemi] != 0],
            hemis=hemi, subject=self.mne_subject,
            subjects_dir=self.subjects_dir)

        # pos_mni = np.vstack(all_mni)
        np.save(fsave.with_suffix(".npy"), label_mni)

        data = np.hstack(data)
        stc = mne.SourceEstimate(
            data, vertices, tmin=0, tstep=0.001, subject=self.mne_subject)
        stc_to_nifti(
            stc, self.fwd, self.mne_subject,
            self.subjects_dir, fsave)

    def save_manual_detections(self):
        pass


def simulate_raw_fast(seconds: int = 2, sampling_freq: float = 200.,
                      n_channels: int = 306):
    ch_names = [f'MEG{n:03}' for n in range(1, n_channels + 1)]
    ch_types = ['mag', 'grad', 'grad'] * 102
    info = mne.create_info(
        ch_names, ch_types=ch_types, sfreq=sampling_freq)

    esfreq = 360
    if sampling_freq < esfreq:
        data = electrocardiogram()[:int(seconds*esfreq)]
        data = signal.resample(data, int(seconds*sampling_freq))
        # plt.plot(
        #     np.linspace(0, 2, 2*esfreq), data[:2*esfreq], 'go-',
        #     np.linspace(0, 2, 2*sfreq), data2[:2*sfreq], '.-')
    else:
        data = electrocardiogram()[:int(seconds*sampling_freq)]

    raw_data = np.repeat(np.array([data]), n_channels, axis=0)
    noise = np.random.normal(0, .1, raw_data.shape)
    raw_data = 1e-9 * (raw_data + noise)
    raw = mne.io.RawArray(raw_data, info)
    del raw_data
    return raw, data
