import cmath
import numpy as np

def discrete_fourier(v, sr):
    n = len(v)
    frequencies = np.zeros(n)
    m = np.zeros((n, n), dtype=complex)
    
    # Complex exponential e^(-2 * pi * i * k * j / n)
    for k in range(n):
        frequencies[k] = frequency(k, sr, n)  # Calculate frequency for each bin
        for j in range(n):
            # Create the complex exponential basis function for DFT
            m[k, j] = cmath.exp(-2j * cmath.pi * k * j / n)
    
    v = np.array(v)
    f = np.dot(m, v)  # Apply the DFT using matrix multiplication
    return complex_arr_magnitude(f[:len(f) // 2]), frequencies[:len(frequencies) // 2]


def fast_fourier(signal, sr):
    frequencies = np.fft.fftfreq(len(signal), d=1/sr)
    fourier_transform = np.fft.fft(signal)
    return fourier_transform, frequencies

def complex_arr_magnitude(a):
    magnitudes = []  # To store the magnitudes
    for i in a:
        magnitudes.append(abs(i))  # Calculate and store magnitude
    return np.array(magnitudes) 

def frequency(k, sr, n):
    return (k * sr) / n  # Frequency for the k-th index