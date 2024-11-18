import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Simulasi data EMG untuk sinyal awal (tidak lelah) dan akhir (lelah)
np.random.seed(0)
time = np.linspace(0, 1, 1000)  # 1 detik sinyal dengan 1000 sample
frequency_before = 50  # frekuensi dominan sebelum aktivitas (tidak lelah)
frequency_after = 20   # frekuensi dominan setelah aktivitas (lelah)

# Sinyal sebelum kelelahan dengan noise
signal_before = np.sin(2 * np.pi * frequency_before * time) + 0.5 * np.random.normal(size=time.shape)

# Sinyal setelah kelelahan dengan noise (frekuensi menurun)
signal_after = np.sin(2 * np.pi * frequency_after * time) + 0.5 * np.random.normal(size=time.shape)

# FFT dari sinyal
fft_before = fft(signal_before)
fft_after = fft(signal_after)
freq = np.fft.fftfreq(len(time), d=1/1000)[:len(time)//2]

# Plotting grafik
fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 8))

# Grafik sinyal EMG - Sebelum Kelelahan
ax1[0].plot(time, signal_before, color='b')
ax1[0].set_title("Sinyal EMG Sebelum Kelelahan")
ax1[0].set_xlabel("Waktu (detik)")
ax1[0].set_ylabel("Amplitudo")

# Grafik sinyal EMG - Setelah Kelelahan
ax1[1].plot(time, signal_after, color='r')
ax1[1].set_title("Sinyal EMG Setelah Kelelahan")
ax1[1].set_xlabel("Waktu (detik)")
ax1[1].set_ylabel("Amplitudo")

# Grafik FFT - Sebelum Kelelahan
ax2[0].plot(freq, np.abs(fft_before)[:len(freq)], color='b')
ax2[0].set_title("Spektrum Frekuensi Sebelum Kelelahan")
ax2[0].set_xlabel("Frekuensi (Hz)")
ax2[0].set_ylabel("Amplitudo")
ax2[0].set_xlim(0, 100)

# Grafik FFT - Setelah Kelelahan
ax2[1].plot(freq, np.abs(fft_after)[:len(freq)], color='r')
ax2[1].set_title("Spektrum Frekuensi Setelah Kelelahan")
ax2[1].set_xlabel("Frekuensi (Hz)")
ax2[1].set_ylabel("Amplitudo")
ax2[1].set_xlim(0, 100)

plt.tight_layout()
plt.show()
