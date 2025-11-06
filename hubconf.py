"""torch.hub configuration."""

dependencies = ["torch", "s3prl"]

import os
import torch        # pylint: disable=wrong-import-position

from singmos.ssl_mos.singmos_v1 import MOS_Predictor as SingMOS_v1 # pylint: disable=wrong-import-position
from singmos.ssl_mos.singmos_pro import MOS_Predictor as SingMOS_Pro


URLS = {
    "singmos_v1": "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_s3prl/ft_wav2vec2_base_960_23steps.pt",
    "singmos_pro": "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_v3/ft_wav2vec2_large_ll60k_mdf_p1_200epochs_all_192epochs.pth",
}
# [Origin]
# "singing_ssl_mos" is derived from official nii-yamagishilab/mos-finetune-ssl, under BSD 3-Clause License (Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics, https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/LICENSE
#  trained on SingMOS dataset (https://huggingface.co/datasets/TangRain/SingMOS)


def singmos_v1(pretrained: bool = True, **kwargs) -> SingMOS_v1:
    """
    `SingMOS_v1` singing naturalness MOS predictor.

    Args:
        pretrained - If True, use URL model; if False, use local model path
        model_path - Local model checkpoint path (required when pretrained=False)
        progress - Whether to show model checkpoint load progress
    """
    base_model_type = "wav2vec2_base_960"
    model = SingMOS_v1(
        ssl_model_type=base_model_type,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained is True:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["singmos_v1"], map_location=device)
    else:
        model_path = kwargs.get('model_path')
        if model_path is None:
            raise ValueError("When pretrained=False, please provide a valid model_path.")
        state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def singmos_pro(pretrained: bool = True, **kwargs) -> SingMOS_Pro:
    """
    `SingMOS_Pro` singing naturalness MOS predictor.

    Args:
        pretrained - If True, use URL model; if False, use local model path
        model_path - Local model checkpoint path (required when pretrained=False)
        progress - Whether to show model checkpoint load progress
    """
    base_model_type = "wav2vec2_large_ll60k"
    model = SingMOS_Pro(
        ssl_model_type=base_model_type,
        use_domain_id=True,
        domain_num=6,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if pretrained is True:
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["singmos_pro"], map_location=device)
    else:
        model_path = kwargs.get('model_path')
        if model_path is None:
            raise ValueError("When pretrained=False, please provide a valid model_path.")
        state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

