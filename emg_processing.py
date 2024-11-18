import numpy as np
import matplotlib.pyplot as plt

# Set parameters for the simulated EMG signal
sampling_rate = 1000  # Hz
time = np.linspace(0, 1, sampling_rate)
frequency_main = 50   # Main frequency in Hz (simulating main muscle frequency)
noise_frequency = 300 # Noise frequency in Hz
amplitude_main = 1.0
amplitude_noise = 0.5

# Generate a sample EMG signal with noise
signal_main = amplitude_main * np.sin(2 * np.pi * frequency_main * time)
signal_noise = amplitude_noise * np.sin(2 * np.pi * noise_frequency * time)
signal_emg = signal_main + signal_noise

# Perform FFT
fft_data = np.fft.fft(signal_emg)
frequencies = np.fft.fftfreq(len(fft_data), d=1/sampling_rate)

# Filter FFT - Removing noise frequencies outside of [20, 150] Hz
fft_filtered = fft_data.copy()
low_cutoff = 20
high_cutoff = 150

# Apply filter
fft_filtered[(frequencies < low_cutoff) | (frequencies > high_cutoff)] = 0

# Inverse FFT to get the filtered signal back to time domain
filtered_signal_emg = np.fft.ifft(fft_filtered)

# Plot the results
fig, axs = plt.subplots(3, 1, figsize=(12, 12))
fig.suptitle('EMG Signal Processing with FFT and Filter', fontsize=16)

# 1. Original Signal in Time Domain
axs[0].plot(time, signal_emg, label="Original EMG Signal")
axs[0].set_title("1. Original Signal (Time Domain)")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Amplitude")
axs[0].legend()

# 2. FFT of Original Signal
axs[1].plot(frequencies[:sampling_rate//2], np.abs(fft_data[:sampling_rate//2]), label="FFT without Filter")
axs[1].set_title("2. FFT of Original Signal (Frequency Domain)")
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()

# 3. FFT of Filtered Signal
axs[2].plot(frequencies[:sampling_rate//2], np.abs(fft_filtered[:sampling_rate//2]), color='orange', label="FFT with Filter")
axs[2].set_title("3. FFT After Applying Frequency Filter")
axs[2].set_xlabel("Frequency (Hz)")
axs[2].set_ylabel("Amplitude")
axs[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
