import torch
import torch.nn as nn

from clinical_ts.models.ecg_foundation_models.ecg_jepa.ecg_jepa import ecg_jepa


def load_encoder(ckpt_dir):
    params = {
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 16,
        "predictor_embed_dim": 384,
        "predictor_depth": 6,
        "predictor_num_heads": 12,
        "c": 8,
        "pos_type": "sincos",
        "mask_scale": (0, 0),
        "leads": list(range(8))
    }
    encoder = ecg_jepa(**params).encoder
    ckpt = torch.load(ckpt_dir)
    encoder.load_state_dict(ckpt["encoder"])
    embed_dim = 768

    return encoder, embed_dim
