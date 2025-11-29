<div align="center">

# SingMOS

</div>

Singing MOS Predictor: A predictor for singing mean-opinion-score prediction.

Our paper link: [SingMOS-Pro: A Comprehensive Benchmark for Singing Quality Assessment](https://arxiv.org/abs/2510.01812)

## Predictors
The SingMOS repository provides an easy-to-use way to perform singing voice MOS prediction.

Currently we provide below models:

| Model        | specifier        | Train Data        | Backbone Model|paper                         |
|--------------|------------------|----------------|----------------|-------------------------------|
| Singing-SSL-MOS | `singmos_pro` | SingMOS-Pro | [wav2vec2_large_ll60k](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav3vec2-large-ll60k) |[Tang (2025)](https://arxiv.org/abs/2510.01812) |
| Singing-SSL-MOS | `singmos_v1` | SingMOS-v1 | [wav2vec2-base-960](https://s3prl.github.io/s3prl/tutorial/upstream_collection.html#wav2vec2-base-960) | [Tang (2024)](https://arxiv.org/abs/2406.10911) |


- `singmos_pro`: Benchmark for Singing MOS Prediction: train a ssl-mos model in [South-Twilight/SingMOS-Predictor](https://github.com/South-Twilight/SingMOS-Predictor) repository using [SingMOS-Pro](https://huggingface.co/datasets/TangRain/SingMOS-Pro) dataset.
- `singmos_v1`: Baseline for [Singing Track in VoiceMOS Challenge 2024](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024): train a ssl-mos model in [nii-yamagishilab/mos-finetune-ssl](https://github.com/nii-yamagishilab/mos-finetune-ssl) repository using [SingMOS-v1](https://huggingface.co/datasets/TangRain/SingMOS-v1) dataset.

**All models were trained at a 16 kHz sampling rate**. 

## News:

- **[2025.11.29]**: Release *SingMOS:v1.1.2* version, fix README.
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

wave, sr = librosa.load("your_audio.wav", sr=None, mono=True)

# if sample rate != 16000, resample the wave.
if sr != 16000:
    wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
    sr = 16000

wave = torch.from_numpy(wave).unsqueeze(0)  # [1, T]
length = torch.tensor([wave.shape[1]], dtype=torch.long)  # [1]

predictor = torch.hub.load("South-Twilight/SingMOS:v1.1.2", "singmos_pro", trust_repo=True)

with torch.no_grad():
    score = predictor(wave, length)

print(f"Pred MOS: {score.item():.4f}")
```

## How to Use
SingMOS use `torch.hub` built-in model loader, so no needs of library import😉  
(As general dependencies, SingMOS requires Python=>3.8, `torch`, `librosa` and `s3prl`.)  

First, instantiate a MOS predictor with model specifier string:
```python
import torch
predictor = torch.hub.load("South-Twilight/SingMOS:v1.1.2", "specifier>", trust_repo=True)
```

Then, pass tensor of singings : wave in `(Batch, Time)`, length in `(Batch)`:
```python
waves = torch.rand((2, 16000)) # Two clips, each 1 sec (sr=16,000)
lengths = []
for i in range(waves.shape[0]):
    lengths.append(waves[i].shape[0])
lengths = torch.tensor(lengths)
# wave: [2, T], length: [2]
score = predictor(waves, lengths)
# tensor([4.0321, 2.0943])
```

Returned scores :: `(Batch,)` are each singing's predicted MOS.  
If you hope MOS average over singings (e.g. for SVS model evaluation), just average them:
```python
average_score = score.mean().item()
# 2.0632
```

### Acknowlegements <!-- omit in toc -->
- SingMOS-Predictor
  - [repository](https://github.com/South-Twilight/SingMOS-Predictor)
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

