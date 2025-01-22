from Fourier import complex_arr_magnitude, discrete_fourier, fast_fourier
import numpy as np

def STF(a, sr, nSamples, stepSize):
    n = len(a)
    windowLength = 128  # Calculate the length of each time window
    
    # Ensure windowLength is valid and check num_windows calculation
    if windowLength > n:
        raise ValueError("Window length must be smaller than the length of the audio signal.")
    
    num_windows = (n - windowLength) // stepSize + 1  # Number of time windows
    
    # Debugging: print values to check
    # print(f"Signal length: {n}")
    # print(f"Window length: {windowLength}")
    # print(f"Number of windows: {num_windows}")

    # Avoid negative or zero dimensions for the matrix
    if num_windows <= 0 or windowLength // 2 <= 0:
        raise ValueError("Invalid matrix dimensions. Check window length and step size.")
    
    matrix = np.zeros((windowLength // 2, num_windows), dtype=float)  # To store the magnitude spectrum for each time window
    time_steps = []  # To store the corresponding time for each window
    
    for i in range(0, n - windowLength + 1, stepSize):
        # Extract the current time window
        window = a[i:i + windowLength]
        x, freqsX = fast_fourier(window, sr)
        magX_dB = 200 * np.log10(np.maximum(x, 1e-10))  
        matrix[:, i // stepSize] = magX_dB[:windowLength // 2]  # Only keep positive frequencies
        time_steps.append(i / sr)  # Append the time corresponding to the start of the window

    return matrix, freqsX, np.array(time_steps)
