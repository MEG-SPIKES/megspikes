import re
import warnings
from pathlib import Path
from typing import List, Union

import mne
import pandas as pd

mne.set_log_level("ERROR")


class CaseManager():
    def __init__(self, root: Union[str, Path, None] = None,
                 case: Union[str, None] = None,
                 free_surfer: Union[str, Path, None] = None):
        """
        Create folders and paths for further operations.

        Parameters
        ----------
        root : str
            the folder with all cases
        case : str
            the name of the case

        """
        self.case = case
        self.root = Path(root)
        self.freesurfer_dir = free_surfer
        # Read case info (Excel)
        self.read_case_info()
        self.dataset = self.case_meg / f"{case}_alphacsc_results.nc"
        self.cluster_dataset = (
            self.case_meg / f"{case}_alphacsc_cluster_results.nc")
        self.manual_cluster_dataset = (
            self.case_meg / f"{case}_manual_cluster_results.nc")

    def read_case_info(self):
        case = self.case
        if not (self.root / 'case_info.xlsx').is_file():
            raise RuntimeError("No case_info.xlsx")
        cases_info = pd.read_excel(self.root / 'case_info.xlsx')
        cond = cases_info['Case name'] == case
        self.recording = cases_info.loc[cond, 'Recording'].values[0]
        self.run = Path(cases_info.loc[cond, 'Files'].values[0]).stem
        self.manual_spikes_fs = cases_info.loc[
            cond, "Manual spikes first sample"].values[0]
        bad_annot_ = cases_info.loc[
            cond, "Bad time periods in the data"].values[0]

        bad_ica = cases_info.loc[cond, "Bad components"].values[0]

        if isinstance(bad_ica, str) or isinstance(bad_ica, int):
            if isinstance(bad_ica, str):
                if len(bad_ica.split(',')) > 1:
                    self.bad_ica_components = [
                        int(i) for i in bad_ica.split(',')]
                elif len(re.findall(r'\d*\.?\d+', bad_ica)) > 0:
                    self.bad_ica_components = [
                        int(re.findall(r'\d*\.?\d+', bad_ica)[0])]
            if isinstance(bad_ica, int):
                self.bad_ica_components = [bad_ica]

        else:
            self.bad_ica_components = []

        if not pd.isna(self.recording):
            self.case_meg = self.root / case / self.recording
        else:
            self.case_meg = self.root / case
            self.recording = ''
        if not pd.isna(bad_annot_):
            bad_annot = {}
            bad_annot["onsets"] = []
            bad_annot["durations"] = []
            bad_annot["descriptions"] = []
            for i in bad_annot_.split(','):
                onset = float(i.split("-")[0])
                end = float(i.split("-")[1])
                bad_annot["onsets"].append(onset)
                bad_annot["durations"].append(end - onset)
                bad_annot["descriptions"].append("BAD")
            self.bad_annot = bad_annot
        else:
            self.bad_annot = False

    def set_basic_folders(self):
        self.case_meg.parent.mkdir(exist_ok=True)
        self.case_meg.mkdir(exist_ok=True)
        root = self.case_meg
        basic_folders = {}
        folders = [
            "MEG_data", "PSD",  "MRI", "MANUAL", "forward_model", "REPORTS"]
        for folder in folders:
            folder_ = root / folder
            if not folder_.is_dir():
                folder_.mkdir()
            basic_folders[folder] = folder_

        # tsss_mc
        basic_folders["tsss_fif_files"] = basic_folders["MEG_data"] / "tsss_mc"
        if not basic_folders["tsss_fif_files"].is_dir():
            basic_folders["tsss_fif_files"].mkdir()

        # tsss_mc_artefact_correction
        basic_folders["art_cor_fif_files"] = basic_folders["MEG_data"] / \
            "tsss_mc_artefact_correction"
        if not basic_folders["art_cor_fif_files"].is_dir():
            basic_folders["art_cor_fif_files"].mkdir()

        # resection
        basic_folders['resection mask'] = basic_folders["MRI"] / \
            "resection.nii"

        # If not post MRI
        if not basic_folders['resection mask'].is_file():
            basic_folders['resection mask'] = basic_folders["MRI"] / \
                "resection.txt"

        # T1 post mni reslice
        basic_folders['mri post'] = basic_folders["MRI"] / \
            "T1_post_mni_reslice.nii"

        # Pre MRI
        basic_folders['mri pre'] = basic_folders["MRI"] / 'T1_pre.nii'

        # Empty room
        basic_folders["empty_room"] = basic_folders["MEG_data"] / "empty_room"
        if basic_folders["empty_room"].is_dir():
            if not sorted(basic_folders["empty_room"].glob('*.fif')) == []:
                basic_folders["empty_room_recording"] = sorted(
                    basic_folders["empty_room"].glob('*.fif'))[0]
            else:
                basic_folders["empty_room_recording"] = None
        else:
            basic_folders["empty_room"].mkdir()
            basic_folders["empty_room_recording"] = None

        self.basic_folders = basic_folders

    def select_fif_file(self, run):
        art_cor = self.basic_folders["art_cor_fif_files"]
        tsss = self.basic_folders["tsss_fif_files"]
        if sorted(art_cor.glob("*{}*".format(run))) != []:
            self.fif_file = sorted(art_cor.glob("*{}*".format(run)))[0]
        else:
            raise RuntimeError("No fif file for analysis")

        if sorted(tsss.glob("*{}*".format(run))) != []:
            self.tsss_file = sorted(tsss.glob("*{}*".format(run)))[0]
        else:
            warnings.warn("No tsss fif file")

    def prepare_forward_model(self, spacings: List[str] = ['ico5', 'oct5'],
                              sensors: Union[str, bool] = True) -> None:
        info = mne.io.read_info(self.fif_file)
        self.info = mne.pick_info(info, mne.pick_types(info, meg=sensors))
        self.fwd = {}
        self.bem, self.src, self.trans = {}, {}, {}
        for spacing in spacings:
            fwd_name = self.basic_folders['forward_model']
            fwd_name = fwd_name / f'forward_{spacing}.fif'
            fwd, bem, src, trans = self._prepare_forward_model(
                fwd_name, self.info, spacing=spacing, n_jobs=7, fixed=False)

            if isinstance(sensors, str):
                fwd = mne.pick_types_forward(fwd, meg=sensors)

            self.fwd[spacing] = fwd
            self.bem[spacing] = bem
            self.src[spacing] = src
            self.trans[spacing] = trans

    def _prepare_forward_model(self, fwd_name, info, spacing='ico5',
                               n_jobs=7, fixed=False):
        """Make forwad solution

        NOTE: Coregistration was done in Brainstorm and affine
        from MRI srucute in BrainStorm was used

        Parameters
        ----------
        freesurfer_dir : str
            FreeSurfer folder (including bem)
        spacing : str, optional
            'oct5' - 1026*2 sources, by default 'ico5' - 10242*2 sources
        """
        fsrc = fwd_name.with_name(f'source_spaces_{spacing}.fif')
        if not fsrc.is_file():
            try:
                src = mne.setup_source_space(
                    self.case, spacing=spacing, add_dist='patch',
                    subjects_dir=self.freesurfer_dir, n_jobs=n_jobs)
            except Exception:
                warnings.warn(f'Using ico4 instead of {spacing}')
                # traceback.print_exc()
                src = mne.setup_source_space(
                    self.case, spacing='ico4', add_dist='patch',
                    subjects_dir=self.freesurfer_dir, n_jobs=n_jobs)
            mne.write_source_spaces(
                fsrc, src, overwrite=True, verbose='error')
        else:
            src = mne.read_source_spaces(fsrc, verbose='error')

        fbem = fwd_name.with_name('bem_solution.fif')
        if not fbem.is_file():
            # (0.3, 0.006, 0.3)  # for three layers
            conductivity = (0.3, )
            try:
                model = mne.make_bem_model(
                    subject=self.case, ico=5, conductivity=conductivity,
                    subjects_dir=self.freesurfer_dir)
            except Exception:
                warnings.warn('Using ico4 instead of ico5 for BEM model')
                # traceback.print_exc()
                model = mne.make_bem_model(
                    subject=self.case, ico=4, conductivity=conductivity,
                    subjects_dir=self.freesurfer_dir)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(fwd_name.with_name('bem_solution.fif'), bem)
        else:
            bem = mne.read_bem_solution(fbem, verbose='error')

        ftrans = fwd_name.with_name('checked_visually_trans.fif')
        trans = mne.read_trans(ftrans)

        # info = mne.io.read_info(self.fif_path)
        if not fwd_name.is_file():
            fwd = mne.make_forward_solution(
                info, trans=trans, src=src, bem=bem,
                meg=True, eeg=False, mindist=5.0)
            mne.write_forward_solution(fwd_name, fwd, overwrite=True)
        else:
            fwd = mne.read_forward_solution(
                str(fwd_name), verbose='error')  # 306 sensors x 20000 dipoles

        if fixed:
            fwd_fixed = mne.convert_forward_solution(
                fwd, surf_ori=True, force_fixed=True, use_cps=True)
            fwd_fixed_name = str(fwd_name.with_name('fixed_forward.fif'))
            mne.write_forward_solution(
                fwd_fixed_name, fwd_fixed, overwrite=True)
            # sio.savemat(str(fwd_name.with_name('fixed_forward.mat')), {
            #             'fwd_fixed': fwd['sol']['data']})

        return fwd, bem, src, trans
