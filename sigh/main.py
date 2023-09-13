#!/usr/bin/env python3
import math
import os
import sys
import time
from typing import Dict, Optional

import fire
import numpy as np
import torch
import whisper
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
    device: str,
    initial_prompt: Optional[str] = None,
    **kwargs: Dict,
) -> Dict:
    pcmf32_pt = torch.tensor(pcmf32, device=device, dtype=torch.float32)
    return model.transcribe(
        audio=pcmf32_pt,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        **kwargs,
    )


def main(
    # AudioAsync params:
    length_ms: int = 5000,
    capture_device_id: int = 1,
    # VAD params:
    vad_thold: float = 0.6,
    freq_thold: float = 80.0,
    # Whisper params:
    model_name: str = "base",
    language: str = "en",
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.2,
    # Prompt params:
    prompt_ms: int = 2000,
    k_prompt: str = "gpt",
    # Stream params:
    keep_ms: int = 100,
    step_ms: int = 500,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  - Whisper device: {device}")
    print(f"  - Whisper model: {model_name}")

    whisper_gen_kwargs = {
        "language": language,
        "beam_size": 5,
        "fp16": True,
        "verbose": None,
        # If the no_speech probability is higher than this value
        # AND the average log probability over sampled tokens
        # is below `logprob_threshold`, consider the segment as silent:
        "logprob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
    }

    model = whisper.load_model(model_name, device=device)

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

    whisper_prompt_tokens = None

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

                    result = transcribe(
                        model,
                        pcmf32_cur,
                        device,
                        initial_prompt=None,
                        **whisper_gen_kwargs,
                    )

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
                    result = transcribe(
                        model,
                        pcmf32,
                        device,
                        initial_prompt=whisper_prompt_tokens,
                        **whisper_gen_kwargs,
                    )
                    txt = result["text"]

                    # print results
                    sys.stdout.write("\33[2K\r")
                    sys.stdout.flush()

                    print(" " * 100, end="")

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
