<div align="center">

# SingMOS

</div>

Singing MOS Predictor (Baseline for [Singing Track in VoiceMOS Challenge 2024](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024)): train a ssl-mos model in [nii-yamagishilab/mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl) repository using [SingMOS](https://huggingface.co/datasets/TangRain/SingMOS) dataset.


Predict subjective score with only 2 lines of code, with various MOS prediction systems.

```python
predictor = torch.hub.load("South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True)
# wave: [B, T], length: [B]
score = predictor(wave, length)
# tensor([3.7730]), good quality singing!
```

## Example
Predict naturalness (Naturalness Mean-Opinion-Score) of your audio by Singing-SSL-MOS:  

```python
import torch
import librosa

wave, sr = librosa.load("<your_audio>.wav", sr=None, mono=True)
predictor = torch.hub.load("South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True)
wave = torch.from_numpy(wave)
length = torch.tensor([wave.shape[1]])
# wave: [B, T], length: [B]
score = predictor(wave, length)
# tensor([3.7730])
```

## How to Use
SingMOS use `torch.hub` built-in model loader, so no needs of library import😉  
(As general dependencies, SingMOS requires Python=>3.8, `torch` and `fairseq`.)  

First, instantiate a MOS predictor with model specifier string:
```python
import torch
predictor = torch.hub.load("South-Twilight/SingMOS:v0.2.0", "<model_specifier>", trust_repo=True)
```

Then, pass tensor of singings : wave in `(Batch, Time)`, length in `(Batch)`:
```python
waves = torch.rand((2, 16000)) # Two clips, each 1 sec (sr=16,000)
length = []
for i in range(waves.shape[0]):
    length.append(waves[i].shape[0])
length = torch.tensor(length)
# wave: [2, T], length: [2]
score = predictor(wave, length)
# tensor([2.0321, 2.0943])
```

Returned scores :: `(Batch,)` are each singing's predicted MOS.  
If you hope MOS average over singings (e.g. for SVS model evaluation), just average them:
```python
average_score = score.mean().item()
# 2.0632
```

## Predictors
This repository is reimplementation collection of various MOS prediction systems.  
Currently we provide below models:  

| Model        | specifier        | paper                         |
|--------------|------------------|-------------------------------|
| Singing-SSL-MOS | `singing_ssl_mos` | [Cooper (2021)][paper_sslmos21] |


### News:

- **[2024.08.28]**: Release *SingMOS:v0.2.0* version to support S3PRL models as base models instead of fairseq models.
- **[2024.06.28]**: Release *SingMOS:v0.1.0* version.

### News:

- **[2024.08.28]**: Release *SingMOS:v0.2.0* version to support S3PRL models as base models instead of fairseq models.
- **[2024.06.28]**: Release *SingMOS:v0.1.0* version.

### Acknowlegements <!-- omit in toc -->
- MOS-Finetune-SSL
  - [paper][paper_sslmos21]
  - [repository](https://github.com/nii-yamagishilab/mos-finetune-ssl)
- SpeechMOS
  - [repository](https://github.com/tarepan/SpeechMOS)

[paper_sslmos21]: https://arxiv.org/abs/2110.02635
