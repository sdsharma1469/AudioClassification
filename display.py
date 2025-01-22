# display.py

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from Fourier import complex_arr_magnitude, discrete_fourier
from WindowFourier import STF
 
def plot_waveform(audio, sr, title="Waveform"):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_spectrogram(S, sr, title="Spectrogram"):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.colorbar(label='Amplitude (dB)')
    plt.title(title)
    plt.show()

def plot_dft(audio, sr, title="Frequency Spectrum"):
    fourier_arr, frequencies_arr = discrete_fourier(audio, sr)
    magnitude_arr = complex_arr_magnitude(fourier_arr)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies_arr, magnitude_arr)  # Plot frequencies vs magnitudes
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

def plot_stft(audio, sr, nSamples=86, stepSize=10, title="STFT Spectrogram"):
    matrix, freqsX, time_steps = STF(audio, sr, nSamples, stepSize)
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(time_steps, freqsX, matrix.T, shading='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()

def display_audio_analysis(audio, sr, nSamples=86, stepSize=10):
    plot_waveform(audio, sr, title="Amplitude-Time Waveform")
    S = librosa.stft(audio, n_fft=2048, hop_length=512)
    plot_spectrogram(S, sr, title="Spectrogram")
    plot_dft(audio, sr, title="Frequency Spectrum")
    plot_stft(audio, sr, nSamples, stepSize, title="STFT Spectrogram")
