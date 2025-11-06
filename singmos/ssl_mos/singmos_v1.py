import os

import torch
import torch.nn as nn

from s3prl.nn import S3PRLUpstream

import random
random.seed(1984)

ssl_model_list = [
    "wavlm_base",
    "wavlm_large",
    "wav2vec2_base_960",
    "wav2vec2_large_ll60k",
    "hubert_base",
    "hubert_large_ll60k",
    "xls_r_300m",
]

def load_ssl_model_s3prl(ssl_model_type, use_proxy = False):
    assert ssl_model_type in ssl_model_list, (
        f"***ERROR***: {ssl_model_type} is not support, please check ssl_model_list."
    )
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


class MOS_Predictor(nn.Module):
    def __init__(
        self,
        ssl_model_type,
        hdim = 128,
        use_lstm = False,
    ) -> None:
        """ MOS Predictor for Singing:
            pitch_num (int): Max range of pitch
                    
        """
        super(MOS_Predictor, self).__init__()

        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)
        self.use_lstm = use_lstm
        if self.use_lstm is True:    
            self.blstm = torch.nn.LSTM(
                input_size = feature_dim, 
                hidden_size = hdim, 
                num_layers = 1, 
                bidirectional=True, 
                batch_first=True
            )
            self.linear = torch.nn.Linear(
                hdim * 2, 1
            )
        else:
            self.linear = torch.nn.Linear(
                feature_dim, 1
            )
        
    
    def forward(
        self,
        audio,
        audio_length,
    ):
        x = self.ssl_model(audio, audio_length)
        # ssl_feature = self.ssl_model(audio)
        
        if self.use_lstm:
            lstm_out, _ = self.blstm(x)
            frame_score = self.linear(lstm_out).squeeze(-1)
            utt_score = frame_score[:, -1]
        else:
            frame_score = self.linear(x).squeeze(-1)
            utt_score = torch.mean(frame_score, dim=1)
        return utt_score
