# coding=utf-8
# Yet another	/\     |   |  |  \   |    /  \
#              /__\    |   |  |   \  |   /    \
#	          /    \   |   |  |   /  |   \    /
#	         /      \  \___/  |__/   |    \__/   Library

# This library is efficient on loading a list of sounds and stores them in RAM.
# WARNING: By loading the sound in RAM, the cache might get deleted in the meantime the sound is playing and fail!

import pyaudio
import numpy as np
import soundfile as sf
import threading
import os
import timeit

class SoundManager:
    def __init__(self):
        self.sound_cache = {}

    # Load sound into memory cache
    def load_sound_to_cache(self, file_path):
        data, sample_rate = sf.read(file_path, dtype='float32')
        self.sound_cache[file_path] = (data, sample_rate)
        print(f"Sound {file_path} loaded and cached in RAM.")

    # Play sound from cache using PyAudio (zero-latency)
    def play_sound_from_cache(self, file_path):
        if file_path not in self.sound_cache:
            raise ValueError(f"Sound {file_path} not found in cache.")

        data, sample_rate = self.sound_cache[file_path]

        # Convert float32 data to int16 PCM format
        pcm_data = (data * 32767).astype(np.int16).tobytes()

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=2 if len(data.shape) == 2 else 1,
                        rate=sample_rate,
                        output=True)

        # Play the sound in chunks to avoid latency
        chunk_size = 1024
        print("Playing sound")
        for i in range(0, len(pcm_data), chunk_size):
            stream.write(pcm_data[i:i + chunk_size])

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Multithreaded loader to load sounds in parallel
    def preload_sounds(self, sound_files):
        threads = []
        for file in sound_files:
            thread = threading.Thread(target=self.load_sound_to_cache, args=(file,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

