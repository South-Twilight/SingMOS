import os
import argparse
import logging

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

from s3prl.nn import S3PRLUpstream

import random
random.seed(1984)

ssl_model_list = [
    "wavlm_base",
    "wavlm_large",
    "wav2vec2_base_960",
    "wav2vec2_large_lv60_cv_swbd_fsh",
    "hubert_base",
    "hubert_large_ll60k",
    "xls_r_300m",
]


def load_ssl_model_s3prl(ssl_model_type, use_proxy = True):
    assert ssl_model_type in ssl_model_list, f"***ERROR***: {ssl_model_type} is not support, please check ssl_model_list."
    if "base" in ssl_model_type:
        SSL_OUT_DIM = 768
    elif "large" in ssl_model_type or ssl_model_type in ["xls_r_300m"]:
        SSL_OUT_DIM = 1024
    if use_proxy:
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    ssl_model = S3PRLUpstream(ssl_model_type)
    return SSL_Model(ssl_model, SSL_OUT_DIM), SSL_OUT_DIM


class SSL_Model(nn.Module):
    def __init__(
        self, 
        ssl_model, 
        ssl_out_dim,
    ):
        super(SSL_Model, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_out_dim = ssl_out_dim
        
    def forward(self, wav, wav_length):
        wav = wav.squeeze(1)  # [B, T]
        ssl_features, ssl_lens = self.ssl_model(wav, wav_length)
        return ssl_features[-1]


class Projection_Layer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        activation_func,
        output_type="scalar",
        mos_clip=True,
    ):
        super(Projection_Layer, self).__init__()

        self.output_type = output_type
        self.mos_clip = mos_clip
        if output_type == "scalar":
            output_dim = 1
            if mos_clip:
                self.proj = nn.Tanh()
        elif output_type == "categorical":
            output_dim = 5
        else:
            raise NotImplementedError("wrong output_type: {}".format(output_type))
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation_func(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        output = self.net(x)
        # mos range clip
        if self.output_type == "scalar" and self.mos_clip:
            return self.proj(output) * 2.0 + 3
        else:
            return output


class MOS_Predictor(nn.Module):
    def __init__(
        self,
        ssl_model_type,
        hdim = 128,
        # projection layer
        activate_func = "relu",
        output_type = "scalar",
        mos_clip = True,
    ) -> None:
        """ MOS Predictor for Singing:
            pitch_num (int): Max range of pitch
        """
        super(MOS_Predictor, self).__init__()

        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)

            # Output layer
        if activate_func == "relu":
            acti_func = nn.ReLU
        else:
            raise NotImplementedError("wrong activate_func: {}".format(activate_func))

        # Mean score projection layer
        self.mean_proj = Projection_Layer(
            in_dim=feature_dim,
            hidden_dim=64,
            activation_func=acti_func,
            output_type=output_type,
            mos_clip=mos_clip,
        )

    
    def forward(
        self,
        audio,
        audio_length,
    ):
        """
        Forward function

        Args:
            audio (torch.Tensor): Wav feature, shape [B, 1, T]
            audio_length (torch.Tensor): Wav length, shape [B]

        Returns:
            dict: Dictionary containing frame and utterance level mean scores, and optionally bias scores if judge_id is provided.
        """
        ssl_feature = self.ssl_model(audio, audio_length)
        # ssl_feature = self.ssl_model(audio)
        T_len = ssl_feature.shape[1]
        
        x = ssl_feature
        # mean net 
        mean_input = x 
        mean_frame_score = self.mean_proj(mean_input)
        mean_utt_score = torch.mean(mean_frame_score, dim=1)
        return mean_utt_score

