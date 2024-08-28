"""torch.hub configuration."""

dependencies = ["torch", "fairseq", "urllib"]

import os
import torch        # pylint: disable=wrong-import-position


from singmos.ssl_mos.ssl_mos import MOS_Predictor, load_ssl_model_s3prl # pylint: disable=wrong-import-position


URLS = {
    "singing_ssl_mos": "https://github.com/South-Twilight/SingMOS/releases/download/checkpoint_s3prl/ft_wav2vec2_base_960_10steps.pt"
}
# [Origin]
# "singing_ssl_mos" is derived from official nii-yamagishilab/mos-finetune-ssl, under BSD 3-Clause License (Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics, https://github.com/nii-yamagishilab/mos-finetune-ssl/blob/main/LICENSE
#  trained on SingMOS dataset (https://huggingface.co/datasets/TangRain/SingMOS)


def download_model(model_url, tgt_path):
    if not os.path.exists(tgt_path):
        os.system(f'wget {model_url} -O {tgt_path}')


def singing_ssl_mos(pretrained: bool = True, **kwargs) -> MOS_Predictor:
    """
    `Singing SSL MOS` singing naturalness MOS predictor.

    Args:
        progress - Whether to show model checkpoint load progress
    """
    if pretrained is True:
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        model_type = "wav2vec2_base_960"
        print("loading base model...")
        ssl_model, ssl_dim = load_ssl_model_s3prl(model_type)
        print("loading base model ended.")

        print("loading ft model...")
        ft_model_path = "checkpoints/ft_wav2vec2_small_10steps.pt"
        download_model(URLS["singing_ssl_mos"], ft_model_path)
        print("loading ft model ended.")

        model = MOS_Predictor(
            ssl_model_type=model_type,
        )
        model.eval()
        model.load_state_dict(torch.load(ft_model_path))
        return model
    else:
        raise ValueError("Please specify pretrained=True and provide a valid model_path.")

