<div align="center">

# SingMOS

</div>

Singing MOS Predictor: A predictor for singing mean-opinion-score prediction.

Our paper link: [SingMOS-Pro: A Comprehensive Benchmark for Singing Quality Assessment](https://arxiv.org/abs/2510.01812)

## Predictors
This repository is reimplementation collection of various MOS prediction systems.  
Currently we provide below models:  

| Model        | specifier        | Train Data        | paper                         |
|--------------|------------------|----------------|-------------------------------|
| Singing-SSL-MOS | `singmos_pro` | SingMOS-Pro | [Tang (2025)](https://arxiv.org/abs/2510.01812) |
| Singing-SSL-MOS | `singmos_v1` | SingMOS-v1 | [Tang (2024)](https://arxiv.org/abs/2406.10911) |


- `singmos_pro`: Benchmark for Singing MOS Prediction: train a ssl-mos model in [South-Twilight/singmos_predictor](https://github.com/South-Twilight/singmos_predictor) repository using [SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro) dataset.
- `singmos_v1`: Baseline for [Singing Track in VoiceMOS Challenge 2024](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024): train a ssl-mos model in [nii-yamagishilab/mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl) repository using [SingMOS-v1](https://huggingface.co/datasets/TangRain/SingMOS-v1) dataset.


## News:

- **[2025.11.11]**: Release *SingMOS:v1.1.1* version, fix bugs with batch inference.
- **[2025.11.06]**: Release *SingMOS:v1.1.0* version, train with SingMOS-Pro.
- **[2025.06.30]**: Release *SingMOS:v0.3.0* version, train with more data.
- **[2024.08.28]**: Release *SingMOS:v0.2.1* version, support S3PRL models as base models instead of fairseq models.
- **[2024.06.28]**: Release *SingMOS:v0.1.0* version.


## Example
Predict naturalness (Naturalness Mean-Opinion-Score) of your audio by Singing-SSL-MOS:  

```python
import torch
import librosa

wave, sr = librosa.load("<your_audio>.wav", sr=None, mono=True)
predictor = torch.hub.load("South-Twilight/SingMOS:v1.1.1", "singmos_pro", trust_repo=True)
wave = torch.from_numpy(wave)
length = torch.tensor([wave.shape[1]])
# wave: [B, T], length: [B]
score = predictor(wave, length)
# tensor([3.7730])
```

## How to Use
SingMOS use `torch.hub` built-in model loader, so no needs of library import😉  
(As general dependencies, SingMOS requires Python=>3.8, `torch` and `s3prl`.)  

First, instantiate a MOS predictor with model specifier string:
```python
import torch
predictor = torch.hub.load("South-Twilight/SingMOS:v1.1.1", "specifier>", trust_repo=True)
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
# tensor([4.0321, 2.0943])
```

Returned scores :: `(Batch,)` are each singing's predicted MOS.  
If you hope MOS average over singings (e.g. for SVS model evaluation), just average them:
```python
average_score = score.mean().item()
# 2.0632
```

### Acknowlegements <!-- omit in toc -->
- MOS-Finetune-SSL
  - [paper][paper_sslmos21]
  - [repository](https://github.com/nii-yamagishilab/mos-finetune-ssl)
- SpeechMOS
  - [repository](https://github.com/tarepan/SpeechMOS)

[paper_sslmos21]: https://arxiv.org/abs/2110.02635

### Citation
```
@misc{tang2025singmosprocomprehensivebenchmarksinging,
      title={SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment}, 
      author={Yuxun Tang and Lan Liu and Wenhao Feng and Yiwen Zhao and Jionghao Han and Yifeng Yu and Jiatong Shi and Qin Jin},
      year={2025},
      eprint={2510.01812},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.01812}, 
}
```

