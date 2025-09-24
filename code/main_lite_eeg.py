"""
EEG-specific Main_Lite_EEG class for the fm-benchmarking project.
"""
from main_lite_base import Main_Lite, FMWrapperBase
from clinical_ts.models.fm_eeg import FMEEG1, FMEEG2

class Main_Lite_EEG(Main_Lite):
    def __init__(self,hparams):
        super().__init__(hparams)
        if hasattr(hparams, 'architecture'):
            if hparams.architecture == "fm_eeg1":
                self.model = FMEEG1(num_classes=1, num_output_tokens=1)
            elif hparams.architecture == "fm_eeg2":
                self.model = FMEEG2(num_classes=1, num_output_tokens=1) 