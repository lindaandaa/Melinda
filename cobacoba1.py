import tkinter as tk
from tkinter import PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Membuat window utama
root = tk.Tk()
root.title("EMG Signal Display")
root.geometry("800x600")
root.configure(bg='#2c3e50')  # Mengatur warna latar GUI

# Menambahkan logo
logo = PhotoImage(file="logo.png")  # Pastikan logo.png ada di folder yang sama
logo_label = tk.Label(root, image=logo, bg='#2c3e50')
logo_label.grid(row=0, column=0, padx=10, pady=10)

# Mengatur warna tombol
start_button = tk.Button(root, text="Start", bg='#16a085', fg='white')
stop_button = tk.Button(root, text="Stop", bg='#e74c3c', fg='white')
start_button.grid(row=1, column=0, padx=5, pady=5)
stop_button.grid(row=1, column=1, padx=5, pady=5)

# Menambahkan judul untuk grafik
title = tk.Label(root, text="EMG Signal and FFT Analysis", font=("Arial", 16), bg='#2c3e50', fg='white')
title.grid(row=0, column=1, padx=10, pady=10)
