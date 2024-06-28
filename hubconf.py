"""torch.hub configuration."""

dependencies = ["torch", "fairseq", "urllib"]

import os
import torch        # pylint: disable=wrong-import-position
import urllib.request

from singmos.ssl_mos.ssl_mos import Singing_SSL_MOS # pylint: disable=wrong-import-position


URLS = {
    "local": "/data3/tyx/mos-finetune-ssl/checkpoints-v0/ckpt_15",
    "singing_ssl_mos": "/data3/tyx/mos-finetune-ssl/checkpoints-v0/ckpt_15"
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
        # if not os.path.exists("checkpoints"):
        #     os.makedirs("checkpoints")
        # model_path = "checkpoints/singing_ssl_mos.pt"
        # download_model(URLS["singing_ssl_mos"], model_path)

        model_path = URLS["local"]
        model = Singing_SSL_MOS(
            model_path=URLS["local"],
            ssl_dim=768,
        )
        model.eval()

        return model
    else:
        raise ValueError("Please specify pretrained=True and provide a valid model_path.")

