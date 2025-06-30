"""torch.hub configuration."""

dependencies = ["torch", "s3prl"]

import os
import torch        # pylint: disable=wrong-import-position

from singmos.ssl_mos.ssl_mos import MOS_Predictor  # pylint: disable=wrong-import-position
from singmos.ssl_mos.ssl_mos2 import MOS_Predictor as MOS_Predictor2


URLS = {
    "singing_ssl_mos": "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_s3prl/ft_wav2vec2_base_960_23steps.pt",
    "singing_ssl_mos_v2": "https://github.com/South-Twilight/SingMOS/releases/download/ckpt_s3prl/ft_wav2vec2_base_960_5steps_full.pt",
}
# [Origin]
# "singing_ssl_mos" is derived from official nii-yamagishilab/mos-finetune-ssl, under BSD 3-Clause License (Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics, https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/LICENSE
#  trained on SingMOS dataset (https://huggingface.co/datasets/TangRain/SingMOS)


def singing_ssl_mos(pretrained: bool = True, **kwargs) -> MOS_Predictor:
    """
    `Singing SSL MOS` singing naturalness MOS predictor.

    Args:
        progress - Whether to show model checkpoint load progress
    """
    if pretrained is True:
        base_model_type = "wav2vec2_base_960"
        model = MOS_Predictor(
            ssl_model_type=base_model_type,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["singing_ssl_mos"], map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        raise ValueError("Please specify pretrained=True and provide a valid model_path.")


def singing_ssl_mos_v2(pretrained: bool = True, **kwargs) -> MOS_Predictor2:
    """
    `Singing SSL MOS2` singing naturalness MOS predictor.

    Args:
        progress - Whether to show model checkpoint load progress
    """
    if pretrained is True:
        base_model_type = "wav2vec2_base_960"
        model = MOS_Predictor2(
            ssl_model_type=base_model_type,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.hub.load_state_dict_from_url(url=URLS["singing_ssl_mos_v2"], map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        raise ValueError("Please specify pretrained=True and provide a valid model_path.")
