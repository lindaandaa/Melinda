#berisi komponen GUI dan menggunakan fungsi-fungsi dari emg_processing.py. 
#Dengan cara ini, file ini fokus hanya pada penanganan tampilan.

import serial
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from collections import deque
from scipy.fft import fft
import scipy.signal as sig
from scipy.interpolate import interp1d
import random
from PIL import ImageDraw, ImageFont
import emg_processing 
import glob
import io

class EGM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoring Electromyographic")
        self.root.geometry("1800x900")
        self.root.configure(bg='aliceblue')

        # Menambahkan judul dan logo kampus
        self.title_frame = tk.Frame(root, bg="white")
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        # Load dan set ukuran logo kampus
        logo_path = "logo_kampus.png"  # Ganti dengan path logo kampus Anda
        self.logo_image = Image.open(logo_path)
        self.logo_image = self.logo_image.resize((70, 70), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)

        # Menampilkan logo kampus di sebelah kiri
        self.logo_label = tk.Label(self.title_frame, image=self.logo_photo, bg="white")
        self.logo_label.pack(side=tk.LEFT, padx=5)

        # Menampilkan judul di tengah
        self.title_label = tk.Label(self.title_frame, text="Monitoring Electromyographic Signals in Human Muscles",
                                    font=("Helvetica", 20, "bold"), bg="white")
        self.title_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.Y)

        # Serial communication setup
        try:
            self.serial_port = serial.Serial('COM6', 9600)  # Sesuaikan dengan port Arduino Anda
        except:
            self.prompt_for_random()
            
        self.serial_connected =False
        self._setup_serial_connection()
        
        #buffer data sinyal EMG dan FFT
        self.data1 = deque(maxlen=2000)   #buffer untuk sinyal emg dalam volt
        self.data2 = deque(maxlen=1000)   #buffer untuk hasil FFT

        self.animation_running = False
        


        # Menu bar
        self.navbar = tk.Menu(root,  bg="alice blue", fg="black", font=("Helvetica", 11, "bold"))
        root.config(menu=self.navbar)

        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=(11))
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Data", command=self.open_csv_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)

        # Options menu
        self.options_menu = tk.Menu(self.navbar, tearoff=0)

        # Save menu
        self.save_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=( 11))
        self.navbar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Data", command=self.save_data)
        self.file_menu.add_separator()
        self.save_menu.add_command(label="Save Image", command=self.save_image)

        # Control Frame (Kanan)
        self.control_frame = ttk.LabelFrame(root, text="Options:")
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # Tombol Start, Stop, Reset
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start)
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(fill=tk.X, pady=5)
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_button.pack(fill=tk.X, pady=5)

        # Axis Control Frame
        self.axis_control_frame = ttk.LabelFrame(self.control_frame, text="Set Axis Range")
        self.axis_control_frame.pack(fill=tk.X, pady=50)

         # Pilihan grafik yang ingin diatur
        self.graph_choice_label = ttk.Label(self.axis_control_frame, text="Choose Graph:")
        self.graph_choice_label.grid(row=0, column=0, padx=5, pady=5)

        self.graph_choice = ttk.Combobox(self.axis_control_frame, values=["Sinyal EMG", "FFT"])
        self.graph_choice.grid(row=0, column=1, padx=5, pady=5)
        self.graph_choice.current(1)  # Set default pilihan ke FFT

        # Pengaturan X min, X max, Y min, Y max
        self.x_min_label = ttk.Label(self.axis_control_frame, text="X Min:")
        self.x_min_label.grid(row=1, column=0, padx=5, pady=5)
        self.x_min_entry = ttk.Entry(self.axis_control_frame)
        self.x_min_entry.grid(row=1, column=1, padx=5, pady=5)

        self.x_max_label = ttk.Label(self.axis_control_frame, text="X Max:")
        self.x_max_label.grid(row=2, column=0, padx=5, pady=5)
        self.x_max_entry = ttk.Entry(self.axis_control_frame)
        self.x_max_entry.grid(row=2, column=1, padx=5, pady=5)

        self.y_min_label = ttk.Label(self.axis_control_frame, text="Y Min:")
        self.y_min_label.grid(row=3, column=0, padx=5, pady=5)
        self.y_min_entry = ttk.Entry(self.axis_control_frame)
        self.y_min_entry.grid(row=3, column=1, padx=5, pady=5)

        self.y_max_label = ttk.Label(self.axis_control_frame, text="Y Max:")
        self.y_max_label.grid(row=4, column=0, padx=5, pady=5)
        self.y_max_entry = ttk.Entry(self.axis_control_frame)
        self.y_max_entry.grid(row=4, column=1, padx=5, pady=5)

        # Tombol untuk mengatur sumbu sesuai dengan input
        self.set_axis_button = ttk.Button(self.axis_control_frame, text="Set Axis", command=self.set_axis_range)
        self.set_axis_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Analysis Frame 
        self.analysis_frame = ttk.LabelFrame(root, text="File Analysis")
        self.analysis_frame.pack(side=tk.RIGHT, pady=10)
        
        # Analysis Frame (Analisis tombol di bawah sumbu)
        self.analysis_button = ttk.Button(self.control_frame, text="Signal Analysis", command=self.calculate_fft_and_mean)
        self.analysis_button.pack(fill=tk.X, pady=5)
        
        
        # Mengatur gaya tombol
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 14, "bold"), background="darkgreen", foreground="black")

         # Footer frame
        self.footer_frame = ttk.Frame(root)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, expand = True)

        self.status_frame = ttk.LabelFrame(self.footer_frame, text="Status", borderwidth=2, relief="groove")
        self.status_frame.pack(side=tk.LEFT, padx=300, pady=5)
        
        self.result_label = ttk.Label(self.status_frame, text="Nilai: N/A", anchor="e")
        self.result_label.pack(side=tk.TOP, padx=10)
        
        self.status_label = ttk.Label(self.status_frame, text="Status Kondisi: Tidak diketahui", anchor="w", font=("Helvetica", 9))
        self.status_label.pack(side=tk.TOP, padx=10)
        
        
         # Frame untuk grafik
        self.graph_frame = tk.Frame(root, bg="mintcream")
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        
        # Membuat Figure dengan ukuran lebih besar
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(25,14 ))  # Sesuaikan ukuran sesuai kebutuhan
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)

        
        # Plot pertama
        self.ax1.set_title('Electromyography Signal')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Voltage')
        self.line1, = self.ax1.plot([], [], lw=2, color='royalblue')
        
        # Plot kedua
        self.ax2.set_title('FFT')
        self.ax2.set_xlabel('Frekuensi (Hz)')
        self.ax2.set_ylabel('Amplitudo (A)')
        self.line2, = self.ax2.plot([], [], lw=2,color='royalblue')
        
        #Ganti Background line chart 1
        self.ax1.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax1.legend()
        #Ganti Background chart2
        self.ax2.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax2.legend()
        # Canvas untuk menampilkan Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

     

        # Automatic save interval
        self.save_interval_ms = 3 * 60 * 1000  # Save data every 3 minutes
        self.root.after(self.save_interval_ms, self.periodic_save)




