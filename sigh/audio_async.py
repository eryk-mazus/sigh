import numpy as np
import pyaudio


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
