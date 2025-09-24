"""
Movement-specific Main_Lite_Movement class for the fm-benchmarking project.
"""
from main_lite_base import Main_Lite, FMWrapperBase
from clinical_ts.models.fm_movement import FMMovement1, FMMovement2

class Main_Lite_Movement(Main_Lite):
    def __init__(self,hparams):
        super().__init__(hparams)
        if hasattr(hparams, 'architecture'):
            if hparams.architecture == "fm_movement1":
                self.model = FMMovement1(num_classes=1, num_output_tokens=1)
            elif hparams.architecture == "fm_movement2":
                self.model = FMMovement2(num_classes=1, num_output_tokens=1) 