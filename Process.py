import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Fourier import discrete_fourier
from WindowFourier import STF
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
from model import MyConv2D
warnings.filterwarnings('ignore')

# Initialize DataFrame
df = pd.DataFrame(columns=["Label", "Audio Length", "Audio Sample", "Spectrogram"])

# Count total audio files for progress bar
total_files = sum(
    len(files)
    for _, _, files in os.walk("test")
    if any(file.endswith((".wav", ".ogg", ".mp3")) for file in files)
)

# Process audio files in "test" directory with progress bar
with tqdm(total=total_files, desc="Processing Audio Files") as pbar:
    for dir in os.listdir("test"):
        dir_path = os.path.join("test", dir)
        if os.path.isdir(dir_path):  # Ensure it's a directory
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if file.endswith((".wav", ".ogg", ".mp3")):  # Only process audio files
                    audio, sr = librosa.load(file_path, sr=None)  # Load audio
                    audio_length = len(audio)  # Length of audio array
                    if audio_length <= 0:
                        pbar.update(1)
                        continue

                    # Compute STFT and mel spectrogram
                    stft_matrix = librosa.stft(audio, n_fft=2048, hop_length=512)
                    S = librosa.feature.melspectrogram(S=np.abs(stft_matrix) ** 2, sr=sr, n_mels=24, fmax=40000)
                    S_dB = librosa.power_to_db(S, ref=np.mean)

                    # Append data to DataFrame
                    df = df._append(
                        {
                            "Label": file,
                            "Audio Length": audio_length,
                            "Audio Sample": audio,
                            "Spectrogram": S_dB
                        },
                        ignore_index=True,
                    )
                    pbar.update(1)

# Optionally, display the first spectrogram
if not df.empty:
    first_spectrogram = df.iloc[0]["Spectrogram"]
    print(first_spectrogram.shape)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(first_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram of {df.iloc[0]["Label"]}')
    plt.tight_layout()
    plt.show()
