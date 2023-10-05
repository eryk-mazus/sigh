import numpy as np
import torch

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=True,
    verbose=False,
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils


def vad(data: np.array, sampling_rate: int = 16_000, threshold: float = 0.3):
    speech_timestamps = get_speech_timestamps(
        data, vad_model, sampling_rate=sampling_rate, threshold=threshold
    )
    return True if speech_timestamps else False
