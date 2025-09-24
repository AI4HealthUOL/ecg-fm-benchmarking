"""
Dummy FMEEG classes for EEG foundation model, derived from FMWrapperBase.
"""
from main_lite_base import FMWrapperBase

class FMEEG1(FMWrapperBase):
    def __init__(self, num_classes=1, num_output_tokens=1):
        super().__init__(num_classes, num_output_tokens)
        # Dummy implementation

class FMEEG2(FMWrapperBase):
    def __init__(self, num_classes=1, num_output_tokens=1):
        super().__init__(num_classes, num_output_tokens)
        # Dummy implementation 