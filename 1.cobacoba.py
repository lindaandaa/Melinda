import numpy as np
import matplotlib.pyplot as plt

# Misalkan 'signal' adalah data sinyal mentah Anda yang telah diambil
# Sampling rate (frekuensi sampling)
fs = 1000  # misal 1000 Hz, sesuaikan dengan data Anda

# Proses FFT pada sinyal mentah
fft_result = np.fft.fft(signal)
fft_magnitude = np.abs(fft_result)  # Ambil magnitudo FFT
frequencies = np.fft.fftfreq(len(signal), d=1/fs)  # Array frekuensi

# Hanya ambil frekuensi positif
positive_frequencies = frequencies[:len(frequencies)//2]
positive_magnitude = fft_magnitude[:len(frequencies)//2]

# Plot hasil FFT dari sinyal mentah
plt.figure(figsize=(10, 5))
plt.plot(positive_frequencies, positive_magnitude, color='blue')
plt.title("FFT dari Sinyal Mentah (Tanpa Filter)")
plt.xlabel("Frekuensi (Hz)")
plt.ylabel("Amplitudo")
plt.grid(True)
plt.show()
