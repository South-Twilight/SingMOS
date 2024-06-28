"""torch.hub configuration."""

dependencies = ["torch", "fairseq", "urllib"]

import os
import torch        # pylint: disable=wrong-import-position
import urllib.request

from singmos.ssl_mos.ssl_mos import Singing_SSL_MOS # pylint: disable=wrong-import-position


URLS = {
    "wav2vec2_small": "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
    "singing_ssl_mos": "https://github.com/South-Twilight/SingMOS/releases/download/checkpoint/ft_wav2vec2_small_15steps.pt"
}
# [Origin]
# "singing_ssl_mos" is derived from official nii-yamagishilab/mos-finetune-ssl, under BSD 3-Clause License (Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics, https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/LICENSE
#  trained on SingMOS dataset (https://huggingface.co/datasets/TangRain/SingMOS)


def download_model(model_url, tgt_path):
    if not os.path.exists(tgt_path):
        urllib.request.urlretrieve(model_url, tgt_path)


def singing_ssl_mos(pretrained: bool = True, **kwargs) -> Singing_SSL_MOS:
    """
    `Singing SSL MOS` singing naturalness MOS predictor.

    Args:
        progress - Whether to show model checkpoint load progress
    """
    if pretrained is True:
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        base_model_path = "checkpoints/wav2vec_small.pt"
        ft_model_path = "checkpoints/ft_wav2vec2_small_15steps.pt"
        download_model(URLS["wav2vec2_small"], base_model_path)
        download_model(URLS["singing_ssl_mos"], ft_model_path)

        model = Singing_SSL_MOS(
            model_path=base_model_path,
        )
        model.eval()
        model.load_state_dict(torch.load(ft_model_path))
        return model
    else:
        raise ValueError("Please specify pretrained=True and provide a valid model_path.")

