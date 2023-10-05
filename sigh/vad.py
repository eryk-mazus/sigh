import math

import numpy as np
from numba import jit


@jit(nopython=True)
def high_pass_filter(data: np.array, cutoff: float, sample_rate: float) -> np.array:
    # python/numpy adaptation of:
    # https://github.com/ggerganov/whisper.cpp/blob/2f52783a080e8955e80e4324fed73e2f906bb80c/examples/common.cpp#L684

    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    y = 0.0
    output = np.empty_like(data)
    output[0] = data[0]

    for i in range(1, len(data)):
        y = alpha * (y + data[i] - data[i - 1])
        output[i] = y

    return output


@jit(nopython=True)
def vad(
    pcmf32,
    sample_rate: int,
    last_ms: int,
    vad_thold: float,
    freq_thold: float,
) -> bool:
    n_samples = pcmf32.size
    n_samples_last = (sample_rate * last_ms) // 1000

    if n_samples_last >= n_samples:
        # not enough samples - assume no speech
        return False

    if freq_thold > 0.0:
        pcmf32 = high_pass_filter(pcmf32, freq_thold, sample_rate)

    energy_all = np.sum(np.abs(pcmf32)) / n_samples
    energy_last = np.sum(np.abs(pcmf32[-n_samples_last:])) / n_samples_last

    return energy_last <= vad_thold * energy_all
