import torch
import numpy as np

from scipy.interpolate import interp1d

import logging



def _convert_to_continuous_f0(f0: np.array) -> np.array:
    if (f0 == 0).all():
        logging.warning("All frames seems to be unvoiced.")
        return f0

    # padding start and end of f0 sequence
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nonzero_idxs = np.where(f0 != 0)[0]

    # perform linear interpolation
    interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
    f0 = interp_fn(np.arange(0, f0.shape[0]))

    return f0


def f0_dio(
    audio,
    sampling_rate,
    hop_size: int =320,
    pitch_min: float=40.0,
    pitch_max: float=799.0,
    use_log_f0: bool=False,
    use_continuous_f0: bool=False,
    use_discrete_f0: bool=False,   # 建议默认 False，避免和 log 冲突
):
    """
    返回 f0: shape=(#frames,), 无声帧为 0。
    - 线性域输出: Hz
    - 对数域输出: log(Hz)
    - 离散输出（若启用）默认在 Hz 上取整；如需半音/音高阶，建议改为 MIDI 量化。
    """
    import pyworld

    # to float64
    if torch.is_tensor(audio):
        x = audio.detach().cpu().numpy().astype(np.float64)
    else:
        x = np.asarray(audio, dtype=np.float64)

    # hop_size(samples) -> frame_period(ms)
    frame_period = 1000.0 * hop_size / float(sampling_rate)

    # DIO + StoneMask
    f0, timeaxis = pyworld.dio(
        x,
        sampling_rate,
        f0_floor=float(pitch_min),
        f0_ceil=float(pitch_max),
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sampling_rate)  # Hz, 无声≈0

    # 连续化（若需要），只处理有声帧
    if use_continuous_f0:
        f0 = _convert_to_continuous_f0(f0)  # 依你已有实现，输出仍在 Hz 域

    # 仅对**有声帧**做裁剪，保持 0 为无声
    voiced = f0 > 0
    # f0[voiced & (f0 < pitch_min)] = pitch_min
    f0[voiced & (f0 > pitch_max)] = pitch_max

    if use_discrete_f0:
        f0[voiced] = np.round(f0[voiced])
        f0[voiced] = np.clip(f0[voiced], pitch_min, pitch_max)

    if use_log_f0:
        f0[voiced] = np.log(f0[voiced])

    return f0


def calc_pitch_histogram(
    audio,
    sampling_rate,
    hop_size: int=320,
    pitch_min: float=40,
    pitch_max: float=799,
    use_log_f0: bool=False,
    use_continuous_f0: bool=False,
    use_discrete_f0: bool=True,
    return_tensor: bool=True,
):
    f0 = f0_dio(
        audio,
        sampling_rate,
        hop_size=hop_size,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        use_log_f0=False,
        use_continuous_f0=False,
        use_discrete_f0=True,
    )
    # 仅使用有声帧
    voiced = f0 > 0
    if not np.any(voiced):
        pitch_histogram = np.zeros(120, dtype=np.float32)
        if return_tensor:
            return torch.tensor(pitch_histogram, dtype=torch.float32)
        return pitch_histogram

    f0_voiced_hz = f0[voiced].astype(np.float64)

    # 将 Hz 转换为 cents（相对于 A4=440Hz）
    reference_freq = 440.0
    f_cent_values = 1200.0 * np.log2(f0_voiced_hz / reference_freq)

    # 量化到 120 个音高类（每 10 cents 一个 bin），取模到 [0, 120)
    i_fcent_values = (f_cent_values / 10.0) % 120.0

    # 统计直方图（向量化）
    num_bins = 120
    bin_indices = i_fcent_values.astype(np.int64)  # 0..119
    pitch_histogram = np.bincount(bin_indices, minlength=num_bins).astype(np.float32)

    # 归一化
    total_frames_with_pitch = bin_indices.size
    if total_frames_with_pitch > 0:
        pitch_histogram /= float(total_frames_with_pitch)

    if return_tensor:
        return torch.tensor(pitch_histogram, dtype=torch.float32)
    return pitch_histogram


def calc_pitch_note(
    audio,
    sampling_rate,
    hop_size: int=320,
    pitch_min: float=40,
    pitch_max: float=799,
    use_log_f0: bool=False,
    use_continuous_f0: bool=False,
    use_discrete_f0: bool=True,
    return_tensor: bool=True,
):
    """Compute MIDI-quantized pitch note sequence and its frame-wise variation.

    Returns:
        pitch_var: (T,) frame-wise difference of MIDI notes (first element copied)
        pitch_note: (T,) MIDI note per frame, unvoiced frames are 0
    """
    f0 = f0_dio(
        audio,
        sampling_rate,
        hop_size=hop_size,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        use_log_f0=False,
        use_continuous_f0=use_continuous_f0,
        use_discrete_f0=False,
    )
    # frame-aligned arrays
    num_frames = f0.shape[0]
    pitch_note = np.zeros(num_frames, dtype=np.float32)

    # voiced mask
    voiced = f0 > 0
    if np.any(voiced):
        f0_voiced_hz = f0[voiced].astype(np.float64)
        # Hz -> MIDI: 69 + 12*log2(f/440)
        midi_vals = 69.0 + 12.0 * np.log2(f0_voiced_hz / 440.0)
        midi_quant = np.rint(midi_vals).astype(np.float32)
        pitch_note[voiced] = midi_quant

    # frame-wise difference (prepend first value)
    if num_frames > 0:
        pitch_var = np.concatenate(([pitch_note[0]], pitch_note[1:] - pitch_note[:-1])).astype(np.float32)
        pitch_var[1:] += 128
    else:
        pitch_var = np.zeros(0, dtype=np.float32)

    if return_tensor:
        pitch_var = torch.tensor(pitch_var, dtype=torch.float32)
        pitch_note = torch.tensor(pitch_note, dtype=torch.float32)
    return pitch_var, pitch_note


def pad_sequence(sequences, max_length=None, padding_value=0, padding_mode="zero_pad"):
    """ 
    Input:
        sequences: List of sequences to pad
        max_length: Maximum length to pad to (if None, use max length in sequences)
        padding_value: Value to use for zero padding (only used when padding_mode="zero_pad")
        padding_mode: "zero_pad" or "repeat"
    Return:
        padded sequences;
    """
    if max_length is None:
        max_length = max(seq.shape[-1] for seq in sequences)
    
    if padding_mode == "zero_pad":
        padded_sequences = torch.full(
            (len(sequences), *sequences[0].shape[:-1], max_length),  
            padding_value,
            dtype=sequences[0].dtype
        )
        for i, seq in enumerate(sequences):
            padded_sequences[i, ..., :seq.shape[-1]] = seq
    elif padding_mode == "repeat":
        padded_sequences = torch.zeros(
            (len(sequences), *sequences[0].shape[:-1], max_length),
            dtype=sequences[0].dtype
        )
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[-1]
            if seq_len > 0:
                # 计算需要重复多少次
                repeat_times = (max_length + seq_len - 1) // seq_len
                # 重复序列
                repeated_seq = seq.repeat(*([1] * (seq.dim() - 1)), repeat_times)
                # 截取到目标长度
                padded_sequences[i, ...] = repeated_seq[..., :max_length]
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}. Must be 'zero_pad' or 'repeat'.")
    
    return padded_sequences


# make_pad_mask and make_non_pad_mask are based on:
# https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/nets_utils.py

def make_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 1],
                 [0, 0, 0, 1]],
                [[0, 0, 1, 1],
                 [0, 0, 1, 1]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_pad_mask(lengths, xs)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_pad_mask(lengths, xs, 1)
        tensor([[[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]],
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1]]], dtype=torch.uint8)
        >>> make_pad_mask(lengths, xs, 2)
        tensor([[[0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 0, 1]],
                [[0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]],
                [[0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1],
                 [0, 0, 1, 1, 1, 1]]], dtype=torch.uint8)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 1, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, xs, length_dim)
