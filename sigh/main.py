#!/usr/bin/env python3
import math
import os
import time

import fire
import numpy as np
import pyaudio
import torch
import whisper
from numba import jit

# avoiding the scipy exit error:
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

WHISPER_SAMPLE_RATE = 16_000


class AudioAsync:
    def __init__(
        self,
        len_ms=2000,  # length of audio buffer in milliseconds, defaults to 2s
        sample_rate=44100,  # number of samples of audio taken per second
        channels=1,  # number of audio channels, 1 - mono, 2 - stereo
        format=pyaudio.paFloat32,
        capture_id: int = 0,  # id of capture device to use
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.frames_per_buffer = int(
            self.sample_rate * len_ms / 1000
        )  # samples per millisecond * len_ms
        self.audio_buffer = np.zeros(self.frames_per_buffer)

        self.p = pyaudio.PyAudio()

        num_devices = self.p.get_host_api_info_by_index(0).get("deviceCount")
        for i in range(num_devices):
            if (
                self.p.get_device_info_by_host_api_device_index(0, i).get(
                    "maxInputChannels"
                )
                > 0
            ):
                device_name = self.p.get_device_info_by_host_api_device_index(0, i).get(
                    "name"
                )
                device_name = device_name.encode("utf-8", errors="replace").decode(
                    "utf-8"
                )
                print(f"  - Capture device #{i}: '{device_name}'")

        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=capture_id,
            stream_callback=self.callback,
        )

    def resume(self):
        self.stream.start_stream()

    def pause(self):
        self.stream.stop_stream()

    def terminate(self):
        self.pause()
        self.stream.close()
        self.p.terminate()

    def clear(self):
        self.audio_buffer = np.zeros(self.frames_per_buffer)

    def callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_buffer = np.roll(self.audio_buffer, -len(audio_data))
        self.audio_buffer[-len(audio_data) :] = audio_data
        return (in_data, pyaudio.paContinue)

    def get(self, ms: int):
        # get audio data from the circular buffer
        num_samples = int(self.sample_rate * ms / 1000)
        return self.audio_buffer[-num_samples:]


@jit(nopython=True)
def high_pass_filter(data: np.array, cutoff: float, sample_rate: float) -> np.array:
    # python adaptation of:
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


def main(
    # AudioAsync params:
    buffer_len_ms: int = 30 * 1000,
    capture_device_id: int = 1,
    # VAD params:
    vad_thold: float = 0.6,
    freq_thold: float = 80.0,
    # Whisper params:
    model_name: str = "medium",
    language: str = "en",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  - Whisper device: {device}")
    print(f"  - Whisper model: {model_name}")

    model = whisper.load_model(model_name, device=device)

    print(f"  - Using capture device: {capture_device_id}")

    audio = AudioAsync(
        len_ms=buffer_len_ms,
        sample_rate=WHISPER_SAMPLE_RATE,
        capture_id=capture_device_id,
    )
    audio.resume()

    # wait for 1 second to avoid any buffered noise
    time.sleep(1)
    audio.clear()

    print("Waiting for voice commands ...")

    try:
        while True:
            time.sleep(100 / 1000)
            data = audio.get(2000)

            if vad(data, WHISPER_SAMPLE_RATE, 1000, vad_thold, freq_thold, False):
                print("Speech detected! Processing ...")

                data = audio.get(3000)
                # detect commands
                data = torch.tensor(data, device=device, dtype=torch.float32)
                result = model.transcribe(
                    audio=data,
                    language=language,
                    beam_size=5,
                    fp16=True,
                    verbose=True,
                    # If the no_speech probability is higher than this value
                    # AND the average log probability over sampled tokens
                    # is below `logprob_threshold`, consider the segment as silent:
                    logprob_threshold=-0.5,
                    no_speech_threshold=0.2,
                )

                print(result)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        audio.terminate()


if __name__ == "__main__":
    fire.Fire(main)


# todo:
# different modes:
# - always transcribe
# - (wake word/prompt) + command
# whisper:
# - cpu

# refactor

# low priority:
# colors
# nicer prints
