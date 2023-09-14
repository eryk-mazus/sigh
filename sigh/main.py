#!/usr/bin/env python3
import math
import os
import sys
import time
from typing import Dict

import fire
import numpy as np
import torch
from faster_whisper import WhisperModel
from rapidfuzz.distance.Levenshtein import normalized_distance

from sigh import AudioAsync

# avoiding the scipy exit error:
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

WHISPER_SAMPLE_RATE = 16_000


def high_pass_filter(data: np.array, cutoff: float, sample_rate: float) -> np.array:
    # python/numpy adaptation of:
    # https://github.com/ggerganov/whisper.cpp/blob/2f52783a080e8955e80e4324fed73e2f906bb80c/examples/common.cpp#L684

    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    y = 0.0
    output = np.empty_like(data)
    output[0] = data[0]

    # TODO:
    # speed up, using numpy:
    for i in range(1, len(data)):
        y = alpha * (y + data[i] - data[i - 1])
        output[i] = y

    return output


def vad(
    pcmf32,
    sample_rate: int,
    last_ms: int,
    vad_thold: float,
    freq_thold: float,
    verbose: bool = False,
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

    if verbose:
        print(
            f"energy_all: {energy_all}, energy_last: {energy_last}, "
            f"vad_thold: {vad_thold}, freq_thold: {freq_thold}"
        )

    return energy_last <= vad_thold * energy_all


def transcribe(
    model,
    pcmf32: np.array,
    **kwargs: Dict,
):
    segments, _ = model.transcribe(pcmf32, **kwargs)
    segments = list(segments)
    return "".join([s.text for s in segments])


def main(
    # AudioAsync params:
    length_ms: int = 5000,
    capture_device_id: int = 1,
    # VAD params:
    vad_thold: float = 0.6,
    freq_thold: float = 80.0,
    # Whisper params:
    model_name: str = "large",
    compute_type: str = "float16",
    language: str = "en",
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.2,
    # Prompt params:
    prompt_ms: int = 2000,
    k_prompt: str = "gpt",
    # Stream params:
    keep_ms: int = 100,
    step_ms: int = 1000,
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  - Whisper device: {device}")
    print(f"  - Whisper model: {model_name}")

    whisper_gen_kwargs = {
        "language": language,
        "beam_size": 5,
        # If the no_speech probability is higher than this value
        # AND the average log probability over sampled tokens
        # is below `logprob_threshold`, consider the segment as silent:
        "no_speech_threshold": no_speech_threshold,
        "log_prob_threshold": log_prob_threshold,
    }

    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print(f"  - Using capture device: {capture_device_id}")

    audio = AudioAsync(
        len_ms=length_ms,
        sample_rate=WHISPER_SAMPLE_RATE,
        capture_id=capture_device_id,
    )
    audio.resume()

    # streaming params:
    keep_ms = min(keep_ms, step_ms)
    length_ms = max(length_ms, step_ms)

    n_samples_len = int((1e-3 * length_ms) * WHISPER_SAMPLE_RATE)
    n_samples_keep = int((1e-3 * keep_ms) * WHISPER_SAMPLE_RATE)

    n_new_line = max(1, length_ms // step_ms - 1)
    print(f"  - # of transcription segments: {n_new_line}")

    pcmf32_old = np.empty(0)

    # prompt:
    have_prompt = True

    n_iter = 0
    # wait for 1 second to avoid any buffered noise
    time.sleep(1)
    audio.clear()

    print("Waiting for command...")

    try:
        while True:
            time.sleep(100 / 1000)
            data = audio.get(2000)

            if not have_prompt:
                if vad(data, WHISPER_SAMPLE_RATE, 1000, vad_thold, freq_thold, False):
                    print("Speech detected! Processing ...")

                    pcmf32_cur = audio.get(prompt_ms)

                    result = transcribe(model, pcmf32_cur, **whisper_gen_kwargs)
                    txt = result["text"].strip().strip(".")

                    if not txt:
                        print("WARNING: prompt not recognized, try again")
                        continue

                    print(f"Heard: {txt}")
                    dist = normalized_distance(txt.lower(), k_prompt)
                    print(f"prompt = {k_prompt}, dist= {dist:.3f}")

                    have_prompt = True
                    audio.clear()
            else:
                print("### Transcription START")
                while True:
                    # process new audio
                    pcmf32_new = audio.get(step_ms)
                    audio.clear()

                    n_samples_new = pcmf32_new.size
                    n_samples_take = min(
                        pcmf32_old.size,
                        max(0, n_samples_keep + n_samples_len - n_samples_new),
                    )

                    pcmf32 = np.hstack((pcmf32_old[-n_samples_take:], pcmf32_new))

                    pcmf32_old = pcmf32

                    # run inference
                    result = transcribe(model, pcmf32, **whisper_gen_kwargs)
                    txt = result

                    # print results
                    sys.stdout.write("\33[2K\r")
                    sys.stdout.flush()

                    print(" " * 200, end="")

                    sys.stdout.write("\33[2K\r")
                    sys.stdout.flush()

                    if txt:
                        print(txt, end="", flush=True)

                    n_iter += 1

                    if n_iter % n_new_line == 0:
                        print("\n", end="", flush=True)

                        # keep part of the audio for next iteration
                        # to try to mitigate word boundary issues
                        pcmf32_old = pcmf32[-n_samples_keep:]

                        # TODO:
                        # add parameter
                        # whisper_prompt_tokens = ""
                        # # Add tokens of the last full length segment as the prompt
                        # # if not no_context (keep context between audio chunks)
                        # whisper_prompt_tokens += txt

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        audio.terminate()


if __name__ == "__main__":
    fire.Fire(main)
