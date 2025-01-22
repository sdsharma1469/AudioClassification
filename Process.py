import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from Fourier import discrete_fourier, complex_arr_magnitude, frequency
import numpy as np
from WindowFourier import STF
import dis

# Initialize DataFrame
df = pd.DataFrame(columns=["Label", "Audio Length", "Audio Sample"])

# Process audio files in "test" directory
for dir in os.listdir("test"):
    dir_path = os.path.join("test", dir)
    if os.path.isdir(dir_path):  # Ensure it's a directory
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if file.endswith((".wav", ".ogg", ".mp3")):  # Only process audio files
                audio, sr = librosa.load(file_path, sr=None)  # Load audio
                audio_length = len(audio)  # Length of audio array
                if audio_length <= 0:
                    continue
                df = df._append(
                    {
                        "Label": file,
                        "Audio Length": audio_length,
                        "Audio Sample": audio,
                    },
                    ignore_index=True,
                )

if not df.empty:
    first_audio = df.iloc[0]["Audio Sample"]
    audio_label = df.iloc[0]["Label"]
    fourier_arr, frequencies_arr = discrete_fourier(first_audio, sr)
    magnitude_arr = fourier_arr
    matrix, freqsX, time_steps = STF(df.iloc[0]["Audio Sample"], sr, nSamples=86, stepSize=10)
    S = librosa.feature.melspectrogram(S=matrix, sr=sr, n_mels=24, fmax=40000)
    S_dB = librosa.power_to_db(S, ref=np.mean)