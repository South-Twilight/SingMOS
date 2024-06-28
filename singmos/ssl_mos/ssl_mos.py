import os
import argparse
import fairseq

import torch
import torchaudio
import torch.nn as nn

import logging

import random
random.seed(1984)

ssl_model_list = [
    "hubert_base",
    "hubert_large",
    "wav2vec2_small",
    "wav2vec2_large",
    "xlsr_base",
]

class SSL_Model(nn.Module):
    def __init__(
        self, 
        model_path, 
        ssl_out_dim=768,
        ssl_model_name = "wav2vec2_small",
    ):
        super(SSL_Model, self).__init__()

        if ssl_model_name in ["hubert_base", "wav2vec2_small"]: 
            ssl_out_dim = 768
        elif ssl_model_name in [ "hubert_large", "wav2vec2_large", "xlsr_base"]:
            ssl_out_dim = 1024
        else:
            raise ValueError("Please check ssl_model_name and make sure in ssl_model_list.")
        
        if ssl_model_name in ssl_model_list:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            ssl_model = model[0]
            ssl_model.remove_pretraining_modules()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        
    def forward(self, wav):
        wav = wav.squeeze(1)  # [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        return x


class Singing_SSL_MOS(nn.Module):
    def __init__(
        self,
        model_path,
        ssl_dim,
        ssl_model_name = "wav2vec2_small",
    ):
        super(Singing_SSL_MOS, self).__init__()
        
        feature_dim = ssl_dim
        self.ssl_model = SSL_Model(
            model_path,
            ssl_dim,
            ssl_model_name,
        )
        self.linear = torch.nn.Linear(
            feature_dim, 1
        )
        
    
    def forward(
        self,
        audio,
    ):
        ssl_feature = self.ssl_model(audio)
        
        T_len = ssl_feature.shape[1]
        
        x = ssl_feature[:, :T_len, :]
        
        pred_score = self.linear(x)
        pred_score = torch.mean(pred_score, dim=1)
        pred_score = pred_score.squeeze(1)
        return pred_score

