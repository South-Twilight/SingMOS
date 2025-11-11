"""
Reference: https://github.com/unilight/sheet
"""

import os
import logging

import torch
import torch.nn as nn

from s3prl.nn import S3PRLUpstream

from .utils import make_non_pad_mask

logger = logging.getLogger()

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
    assert ssl_model_type in ssl_model_list, f"***ERROR***: {ssl_model_type} is not support, please check ssl_model_list."
    if "base" in ssl_model_type:
        SSL_OUT_DIM = 768
    elif "large" in ssl_model_type or ssl_model_type in ["xls_r_300m", "wav2vec2_large_ll60k"]:
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

# Main Module
class MOS_Predictor(nn.Module):
    def __init__(
        self,
        ssl_model_type,
        hdim = 128,
        dnn_dim = 64,
        # domain related
        use_domain_id = False,
        domain_num = 5,
        # pitch related
        use_pitch = False,
        pitch_type = "note",
        pitch_num: int = 120,
        # judge related
        use_judge_id = False,
        judge_num = 50,
        # projection layer
        activate_func = "relu",
        output_type = "scalar",
        mos_clip = True,
        # loss related:
        use_margin = True,
        loss_type = "L1",
        margin = 0.1,
        use_frame_level = False,
        alpha_frame = 0.6,
    ) -> None:
        """ MOS Predictor for Singing:
            pitch_num (int): For histogram bins or embedding size for raw/MIDI.
        """
        super(MOS_Predictor, self).__init__()

        self.ssl_model, feature_dim = load_ssl_model_s3prl(ssl_model_type)

        # Activation
        if activate_func == "relu":
            acti_func = nn.ReLU
        else:
            raise NotImplementedError("wrong activate_func: {}".format(activate_func))

        # Pitch modules
        self.use_pitch = use_pitch
        self.pitch_type = pitch_type
        self.pitch_num = pitch_num

        if self.use_pitch:
            if self.pitch_type == "note":
                # pitch_note: MIDI [0,127] as long idx -> embedding
                self.pitch_note_emb = nn.Embedding(num_embeddings=self.pitch_num, embedding_dim=hdim)
                # pitch_var: MIDI Var [-127, 127] / [0, 255] -> project to hdim
                self.pitch_var_emb = nn.Embedding(num_embeddings=self.pitch_num * 2, embedding_dim=hdim)
                feature_dim += hdim * 2
            elif self.pitch_type == "histogram":
                # 120-bin pitch histogram -> project to hdim, then tile across time
                self.pitch_hist_proj = nn.Linear(self.pitch_num, hdim)
                feature_dim += hdim
            elif self.pitch_type == "raw":
                # raw discrete pitch (e.g., Hz integer-quantized) -> embedding
                self.pitch_emb = nn.Embedding(num_embeddings=self.pitch_num, embedding_dim=hdim)
                feature_dim += hdim
            else:
                raise ValueError(f"Unsupported pitch_type: {self.pitch_type}")
        
        # Judge modules
        self.use_judge_id = use_judge_id
        self.judge_num = judge_num
        if self.use_judge_id:
            self.judge_emb = nn.Embedding(num_embeddings=self.judge_num, embedding_dim=hdim)
            feature_dim += hdim
        
        # Domain modules
        self.use_domain_id = use_domain_id
        self.domain_num = domain_num
        if self.use_domain_id:
            self.domain_emb = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=hdim)
            feature_dim += hdim

        # Output layers
        self.decoder = Projection_Layer(
            in_dim=feature_dim,
            hidden_dim=dnn_dim,
            activation_func=acti_func,
            output_type=output_type,
            mos_clip=mos_clip,
        )

        # Loss setup
        self.use_frame_level = use_frame_level
        self.alpha_frame = alpha_frame

    
    def forward(
        self,
        audio,
        audio_length,
        pitch_var = None,
        pitch_note = None,
        pitch_histogram = None,
        pitch = None,
        judge_id = None,
        domain_id = None,
        # ground truth
        gt_utt_score = None,
        gt_frame_score = None,
        is_train = True,
    ):
        """
        Forward function

        Args:
            audio (torch.Tensor): Wav feature, shape [B, 1, T]
            audio_length (torch.Tensor): Wav length, shape [B]
            pitch_var (torch.Tensor, optional): [B, T]
            pitch_note (torch.Tensor, optional): [B, T]
            pitch_histogram (torch.Tensor, optional): [B, 120] or [B, 120, T]
            pitch (torch.Tensor, optional): [B, T]
        Returns:
            loss, stats, ret_val
        """
        ssl_feature = self.ssl_model(audio, audio_length)  # [B, T, D]
        bs, T_len = ssl_feature.shape[0], ssl_feature.shape[1]
        audio_token_length = torch.ceil(audio_length / 320)

        x = ssl_feature

        if self.use_pitch:
            if self.pitch_type == "note" and pitch_note is not None and pitch_var is not None:
                # pitch_note: float -> clamp to [0,127] and long
                pitch_note_idx = pitch_note.long()
                note_feat = self.pitch_note_emb(pitch_note_idx)  # [B, T, hdim]
                # pitch_var: float -> project
                pitch_var_idx = pitch_var.long()
                var_feat = self.pitch_var_emb(pitch_var_idx)  # [B, T, hdim]
                x = torch.cat((x, note_feat, var_feat), dim=-1)
            elif self.pitch_type == "histogram" and pitch_histogram is not None:
                # Accept [B, 120] or [B, 120, T]
                if pitch_histogram.dim() == 3:
                    # Reduce over time if provided with padding along time
                    hist = pitch_histogram.mean(dim=-1)  # [B, 120]
                else:
                    hist = pitch_histogram  # [B, 120]
                hist_feat = self.pitch_hist_proj(hist)  # [B, hdim]
                hist_feat = hist_feat.unsqueeze(1).expand(-1, T_len, -1)  # tile over time
                x = torch.cat((x, hist_feat), dim=-1)
            elif self.pitch_type == "raw" and pitch is not None:
                # clamp to embedding range
                pitch_idx = pitch.long()
                raw_feat = self.pitch_emb(pitch_idx)  # [B, T, hdim]
                x = torch.cat((x, raw_feat), dim=-1)
        
        if self.use_judge_id:
            if judge_id is None:
                judge_id = torch.Tensor([0] * bs).to(x.device)
            judge_idx = judge_id.long()
            judge_feat = self.judge_emb(judge_idx)  # [B, hdim]
            judge_feat = judge_feat.unsqueeze(1).expand(-1, T_len, -1)  # tile over time
            x = torch.cat((x, judge_feat), dim=-1)
        
        if self.use_domain_id:
            if domain_id is None:
                domain_id = torch.Tensor([1] * bs).to(x.device)
            domain_idx = domain_id.long()
            domain_feat = self.domain_emb(domain_idx) # [B, hdim]
            domain_feat = domain_feat.unsqueeze(1).expand(-1, T_len, -1) # tile over time
            x = torch.cat((x, domain_feat), dim=-1)

        # mean net 
        decoder_input = x 
        pred_frame_score = self.decoder(decoder_input).squeeze(-1)
        pred_utt_score = torch.mean(pred_frame_score, dim=1)

        return pred_utt_score

