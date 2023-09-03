#!/usr/bin/env python3
import os
import math
import time

import numpy as np
import pyaudio
from numba import jit

# preventing ugly scipy exit error:
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"


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
                print(
                    f"  - Capture device #{i}: '{self.p.get_device_info_by_host_api_device_index(0, i).get('name')}'"
                )
        print(f"Using capture device: {capture_id}")

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
def high_pass_filter(data, cutoff: float, sample_rate: float):
    # python adaptation of:
    # https://github.com/ggerganov/whisper.cpp/blob/2f52783a080e8955e80e4324fed73e2f906bb80c/examples/common.cpp#L684
    rc = 1.0 / (2.0 * math.pi * cutoff)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    y = data[0]

    for i in range(1, len(data)):
        y = alpha * (y + data[i] - data[i - 1])
        data[i] = y

    return data


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
            f"energy_all: {energy_all}, energy_last: {energy_last}, vad_thold: {vad_thold}, freq_thold: {freq_thold}"
        )

    return energy_last <= vad_thold * energy_all


if __name__ == "__main__":
    # audio async params:
    len_ms = 30 * 1000

    # vad params:
    WHISPER_SAMPLE_RATE = 16000
    vad_thold = 0.6
    freq_thold = 0.0
    # freq_thold = 80.0
    vad_verbose = False

    # main loop:
    audio = AudioAsync(len_ms=len_ms, capture_id=1)
    audio.resume()

    # wait for 1 second to avoid any buffered noise
    time.sleep(1)
    audio.clear()

    try:
        while True:
            time.sleep(100 / 1000)

            data = audio.get(2000)
            # print(f"Fetched audio: {data}\t\tShape: {data.shape}")

            if vad(data, WHISPER_SAMPLE_RATE, 1000, vad_thold, freq_thold, vad_verbose):
                print("Speech detected! Processing ...")

                # detect commands
                ...

            # else:
            #     print("Nope...")

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        audio.terminate()


# todo:
# add and parse args
# different modes: always transcribe
# prompt + transcribe

# nicer prints
