import os
from PIL import Image, ImageTk
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import serial
from collections import deque
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import scipy.signal as sig
from scipy.interpolate import interp1d
import random
from PIL import ImageDraw, ImageFont


# Paramater filter
lowcut = 10.0  # Batas bawah frekuensi (Hz)
highcut = 500.0  # Batas atas frekuensi (Hz)
fs = 1000.0  # Frekuensi sampling (Hz) menunjukkan seberapa sering data diambil dalam satu detik

# Fungsi Band-Pass Filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Fungsi yang menerapkan band-pass filter untuk sinyal"""
    nyquist = 0.5 * fs  # Frekuensi Nyquist,  setengah dari frekuensi sampling.
    low = lowcut / nyquist # Normalisasi frekuensi batas bawah dengan frekuensi Nyquist
    high = highcut / nyquist
    # Menggunakan fungsi butter() untuk mendesain filter Butterworth
    b, a = butter(order, [low, high], btype='band')
    # Menggunakan fungsi filtfilt() untuk menerapkan filter ke data 
    y = filtfilt(b, a, data)
    return y   # Mengembalikan data yang sudah difilter

class EGM_GUI:
    def __init__(self, root):
        """ Konfigurasi dasar window """
        self.root = root   # Menyimpan referensi ke jendela root tkinter
        self.root.title("Monitoring Electromyographic")   # Mengatur judul jendela GUI
        self.root.geometry("1950x950")           # Mengatur ukuran jendela GUI
        self.root.configure(bg="#ECEFF1")  # Mengatur warna latar belakang
        
        ### Frame untuk judul dan logo ###
        # Membuat frame judul dan menempatkan frame dibagian atas secara horizontal
        self.title_frame = tk.Frame(root, bg="#d3eff2")   # Warna latar belakang judul frame
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Logo (Set dan ukuran)
        logo_path = "Melinda/logo_kampus.png"  # Pastikan logo.png ada di folder yang sama
        self.logo_image = Image.open(logo_path)   #Path file untuk logo kampus
        self.logo_image = self.logo_image.resize((70, 70), Image.LANCZOS)   # Mengubah ukuran gambar logo 
        self.logo_photo = ImageTk.PhotoImage(self.logo_image) # Konfersi gambar sesuai format tkinter
        # Membuat Tampilan Logo di Frame judul sebelah kiri
        self.logo_label = tk.Label(self.title_frame, image=self.logo_photo, bg="#d3eff2")  
        self.logo_label.pack(side=tk.LEFT, padx=5)
        # Membuat Judul dan penampilkan pada bagian atas frame secara center
        self.title_label = tk.Label(self.title_frame, 
                                    text="Monitoring Electromyographic Signals in Human Muscles",
                                    font=("Arial Bold", 28, "bold"), 
                                    bg="#d3eff2", fg="#04142e", pady=15)
        self.title_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.Y, anchor="center")


        """ Style Tombol dengan Efek Hover """ 
        # Membuat objek style untuk tombol
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14), padding=8, background="#81D4FA", foreground="#0D47A1")
        
        # Menambahkan efek perubahan warna pada tombol saat ditekan atau saat aktif
        style.map("TButton",
                foreground=[('pressed', 'darkblue'), ('active', 'darkgreen')],
                background=[('pressed', '!disabled', 'lightblue'), ('active', 'lightgreen')])


        ### Setup Navbar (Menu Utama) ###
        # Membuat dan mengonfigurasi menu utama
        self.navbar = tk.Menu(root,  bg="#0288D1", fg="white", 
                              font=("Arial", 14, "bold"))
        root.config(menu=self.navbar)  # Menetapkan navbar ke window utama
        
        
        ### File Menu (Menu untuk file (Open, Exit)) ###
        # Membuat menu file untuk membuka data dan keluar
        self.file_menu = tk.Menu(self.navbar, tearoff=0, bg="#d3eff2", 
                                 fg="black", font=("Arial", 12))
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Data", command=self.open_csv_file)   # Fungsi untuk membuka file
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)   # Fungsi untuk keluar aplikasi
        
        ### Save Menu (Menu untuk menyimpan data dan gambar) ###
        # Membuat menu save untuk menyimpan data dan gambar yang ditampilkan
        self.save_menu = tk.Menu(self.navbar, tearoff=0, bg="#d3eff2", 
                                 fg="black", font=("Arial", 12))
        self.navbar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Data", command=self.save_data)  # Fungsi untuk menyimpan data
        self.save_menu.add_separator()
        self.save_menu.add_command(label="Save Image", command=self.save_image)  # Fungsi untuk menyimpan gambar


        ### Option Menu (Menu untuk pengaturan) ###
        self.options_menu = tk.Menu(self.navbar, tearoff=0)
        
        # Control Frame (Frame untuk tombol kontrol (Start, Stop, Reset).)
        self.control_frame = ttk.LabelFrame(root, text="Options" )
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=True)
        
        # Tombol Start (Tombol untuk memulai pengambilan data)
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start)
        self.start_button.pack(padx=5, pady=10, fill=tk.X) 
        # Tombol Stop (Tombol untuk menghentikan pengambilan data.)
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(padx=5, pady=10, fill=tk.X)
        # Tombol Reset (Tombol untuk mereset pengaturan atau grafik.)
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_button.pack(padx=5, pady=10, fill=tk.X)


        
        ### Axis Control Frame  (Frame untuk pengaturan sumbu grafik.)###
        self.axis_control_frame = ttk.LabelFrame(self.control_frame, text="Set Axis Range")
        self.axis_control_frame.pack(fill=tk.X, pady=(30,30))   # Kurangi padding bawah
        # Pilihan grafik yang ingin diatur 
        self.graph_choice_label = ttk.Label(self.axis_control_frame, text="Choose Graph:")
        self.graph_choice_label.grid(row=0, column=0, padx=5, pady=15)
        # Pilihan grafik yg tersedia
        self.graph_choice = ttk.Combobox(self.axis_control_frame, values=["Electromyography Signal", "FFT"])
        self.graph_choice.grid(row=0, column=1, padx=5, pady=5)
        self.graph_choice.current(1)  # ComboBox untuk memilih jenis grafik (Set default pilihan ke FFT)
        ### Pengaturan X min, X max, Y min, Y max ( Label untuk pengaturan sumbu) ###
        # Pengaturan x min
        self.x_min_label = ttk.Label(self.axis_control_frame, text="X Min:")
        self.x_min_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.x_min_entry = ttk.Entry(self.axis_control_frame)
        self.x_min_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        # Pengaturan X Max
        self.x_max_label = ttk.Label(self.axis_control_frame, text="X Max:")
        self.x_max_label.grid(row=2, column=0, padx=5, pady=10, sticky="w")
        self.x_max_entry = ttk.Entry(self.axis_control_frame)
        self.x_max_entry.grid(row=2, column=1, padx=5, pady=10, sticky="ew")
        #Pengaturan Y Min
        self.y_min_label = ttk.Label(self.axis_control_frame, text="Y Min:")
        self.y_min_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.y_min_entry = ttk.Entry(self.axis_control_frame)
        self.y_min_entry.grid(row=3, column=1, padx=5, pady=10, sticky="ew")
        #Pengaturan Y Max
        self.y_max_label = ttk.Label(self.axis_control_frame, text="Y Max:")
        self.y_max_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.y_max_entry = ttk.Entry(self.axis_control_frame)
        self.y_max_entry.grid(row=4, column=1, padx=5, pady=10, sticky="ew")
        # Tombol untuk mengatur sumbu sesuai dengan input
        self.set_axis_button = ttk.Button(self.axis_control_frame, text="Set Axis", command=self.set_axis_range)
        self.set_axis_button.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")
        
        
        ### Analysis Button (Tombol untuk Analisis sinyal) ###
        self.anal_control_frame = ttk.LabelFrame(self.control_frame, text="Analysis")
        self.anal_control_frame.pack(side=tk.TOP, fill=tk.X, padx=0, pady=(10, 15), anchor="n") 
        # Analysis Frame (Analisis tombol di bawah sumbu)
        self.analysis_button = ttk.Button(self.anal_control_frame, text="Signal Analysis", 
                                          command=self.calculate_fft_and_mean)
        self.analysis_button.pack(fill=tk.X, pady=10)
        # Mengatur gaya tombol
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 17, "bold"),  background="darkblue", foreground="#024375")
        
        ### Footer Frame (Frame bagian bawah GUI untuk Status informasi dan Kondisi) ###
        footer_frame = ttk.Frame(root, relief="solid", borderwidth=2)  # Membuat frame utama untuk footer
        footer_frame.place(relx=1.0, rely=1.0, anchor="se", width=290, height=150, x=20, y=-13)  # Pojok kanan bawah
        
        ### Condition Frame (Frame untuk kondisi dan status) ###
        condition_frame = ttk.LabelFrame(footer_frame, text="Condition",  padding=(15, 5))
        condition_frame.pack(side=tk.RIGHT, padx=15, pady=5) # Letakkan di kiri dalam footer_frame
        # Label untuk menampilkan nilai Mean Amplitude
        self.mean_amplitude_label = ttk.Label(condition_frame,  text="Mean Amplitude: N/A",   
                                              font=("Arial", 10), foreground="black")
        self.mean_amplitude_label.pack(side=tk.TOP, padx=15, pady=5, fill=tk.X)

        # Label untuk menampilkan nilai Median Frequency
        self.median_frequency_label = ttk.Label(condition_frame, text="Median Frequency: N/A Hz", 
                                                font=("Arial", 10), foreground="black")
        self.median_frequency_label.pack(side=tk.TOP, padx=15, pady=5, fill=tk.X)

        # Label untuk menampilkan status kondisi
        self.status_condition_label = ttk.Label(condition_frame, text="Condition Status: Unknown", 
                                                font=("Arial", 10),  foreground="black")
        self.status_condition_label.pack(side=tk.TOP, padx=15, pady=5, fill=tk.X)
        # Membuat LabelFrame untuk "Kondisi" di sebelah kiri
        condition_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
    
        ### Graph Frame (Frame untuk menampilkan grafik) ###
        self.graph_frame = tk.Frame(root, bg="#E1F5FE")
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        # Membuat Figure dengan ukuran lebih besar
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20,20 ))  # Sesuaikan ukuran sesuai kebutuhan
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        
        ### First Plot (Plot pertama untuk sinyal EMG) ###
        # Mengatur judul untuk ax1 dengan font,dan warna yg disesuaikan
        self.ax1.set_title('Electromyography Signal', fontsize=18, fontweight='bold', color="#0D47A1")
        self.ax1.set_xlabel('Time', fontsize=14, color="#0D47A1" )
        self.ax1.set_ylabel('Voltage', fontsize=14, color="#0D47A1" )
        #Ganti Background line chart 1
        self.line1, = self.ax1.plot([], [], lw=2, color="#406e94")
        self.ax1.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax1.legend()
        
        ### Second Plot (Plot kedua untuk FFT) ###
        # Mengatur judul untuk ax2 dengan font,dan warna yg disesuaikan
        self.ax2.set_title('FFT', fontsize=18,  fontweight='bold', color="#0D47A1")
        self.ax2.set_xlabel('Frequency (Hz)', fontsize=14, color="#0D47A1" )
        self.ax2.set_ylabel('Amplitude (A)', fontsize=14, color="#0D47A1" )
        #Ganti Background chart 2
        self.line2, = self.ax2.plot([], [], lw=2,color= "#2F4F4F")
        self.ax2.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax2.legend()
        
        ### Canvas for Graph (Canvas untuk menampilkan grafik pada GUI) ###
        # Canvas untuk menampilkan Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    
        
        """Serial communication setup"""
        # Menghubungkan ke Arduino pada port COM6 dengan baud rate 9600
        try:
            self.serial_port = serial.Serial('COM6', 9600)  # Sesuaikan dengan port Arduino
        except:
            self.prompt_for_random()  # Jika gagal terhubung, tampilkan prompt untuk memilih port
            
        # Insialisasi data dan status
        self.data1 = deque(maxlen=2000) # Buffer data untuk grafik pertama (sampel EMG)
        self.data2 = deque(maxlen=1000) # Buffer data untuk grafik kedua (FFT)
        self.animation_running = False  # Status animasi (dimulai atau berhenti)
        self.data_buffer = deque(maxlen=1024)  # Buffer untuk data EMG mentah


        # Interval otomatis untuk menyimpan data setiap 3 menit
        self.save_interval_ms = 3 * 60 * 1000  # Interval penyimpanan dalam milidetik
        self.root.after(self.save_interval_ms, self.periodic_save)  # Fungsi penyimpanan otomatis


    def start(self):
        """ Fungsi untuk memulai animasi Pengambilan Data """
        if not self.animation_running:
            self.animation_running = True   # Menandakan bahwa animasi sedang berjalan
            self.animation()   # Memulai animasi pengambilan data

    def stop(self):
        """ Fungsi untuk menghentikan animasi """
        self.animation_running = False  # Menghentikan animasi

    def reset(self):
        """ Membersihkan data, grafik, dan UI """
    
        self.data1.clear()  # Menghapus data grafik pertama
        self.data2.clear()  # Menghapus data grafik kedua
        self.line1.set_data([], [])  # Menghapus grafik pertama
        self.line2.set_data([], [])  # Menghapus grafik kedua
        
        
          # Mengembalikan sumbu grafik ke auto-scaling
        self.ax1.autoscale(enable=True, axis='both', tight=False)
        self.ax2.autoscale(enable=True, axis='both', tight=False)
            
        
        # Reset sumbu grafik dan tampilan grafik
        for ax in [self.ax1, self.ax2]:
            ax.relim()  # Menyesuaikan ulang batas sumbu grafik
            ax.autoscale_view()  # Menyesuaikan tampilan grafik agar sesuai dengan data

        self.canvas.draw()  # Menggambar ulang canvas untuk memperbarui tampilan


         # Reset label kondisi
        self.mean_amplitude_label.config(text="Mean Amplitude: N/A")
        self.median_frequency_label.config(text="Median Frequency: N/A Hz")
        self.status_condition_label.config(text="Condition Status: Unknown")

        # Hapus teks pada entry axis control
        for entry in [self.x_min_entry, self.x_max_entry, self.y_min_entry, self.y_max_entry]:
            entry.delete(0, tk.END)
        
         # Reset ComboBox jika ada
        self.graph_choice.set('FFT')  # Atur ComboBox ke pilihan default
        print("Reset complete.")
        # Pastikan `animation_running` ter-set ke `False` setelah reset
        #self.animation_running = False
        
    
    def animation(self):
        """ Memperbarui sinyal secara berkala jika animasi sedang berjalan """
        def update_plot():
            if self.animation_running:  #Mengecek apakah animasi sedang berjalan
                if self.use_random_data:
                    # Menggunakan data acak jika Arduino tidak terhubung
                    value = random.uniform(0, 5)  # Menghasilkan angka acak antara 0 dan 5
                else:
                    try:
                        # Kode membaca data dari Arduino
                        data = self.serial_port.readline().decode('ascii').strip()
                        if data:
                            value = float(data)  # Mengonversi data yang dibaca menjadi float
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read from Arduino: {e}")
                        return

                # Menambahkan data ke dalam buffer untuk grafik
                self.data1.append(value)
                
                # Menerapkan filter Low-pass Butterworth
                if len(self.data1) > 5:  # Pastikan data cukup sebelum filter
                    b, a = sig.butter(4, 0.1, 'low')    # Membuat filter low-pass
                    filtered_signal = sig.lfilter(b, a, self.data1)  # Menerapkan filter
                    self.data2.append(filtered_signal[-1])     # Menambahkan data yang sudah difilter
                
                # Mengupdate grafik
                self.line1.set_data(range(len(self.data1)), self.data1)
                self.ax1.relim()  # Menyesuaikan ulang sumbu X dan Y
                self.ax1.autoscale_view()  # Menyesuaikan tampilan grafik
                self.canvas.draw()  # Menggambar ulang canvas
                self.canvas.flush_events()  # Mengupdate event di canvas

                


                # Memanggil fungsi update_plot lagi setelah 10ms
                self.root.after(10, update_plot)
        update_plot()   # Memulai update plot
        
                 
        
    def update_data(self):
        # Simulasi membaca data dari Arduino (diganti dengan data asli di implementasi nyata)
        new_data = np.random.normal(0, 1, 100)  # Ganti dengan data asli dari Arduino
        self.data1.extend(new_data)    # Menambahkan data baru ke buffer

        # Menjaga buffer tidak lebih dari 1000 data poin
        if len(self.data1) > 1000:
            self.data1 = self.data1[-1000:]  # Mengambil hanya 1000 data terakhir

        # Panggil fungsi FFT dan analisis setiap kali data diperbarui
        self.calculate_fft_and_mean()

        # Jadwalkan pemanggilan fungsi ini lagi dalam 500 ms
        self.root.after(500, self.update_data)
        
        
    def calculate_fft_and_mean(self):
        """ Gabungan fungsi untuk memperbarui grafik FFT dan menghitung statistik sinyal EMG """
        # Ambil data dari grafik EMG (data waktu)
        y_data_emg = np.array(self.line1.get_ydata())
        # Normalisasi data dengan Min-Max Normalization
        if len(y_data_emg) > 0:
            y_data_emg = (y_data_emg - np.min(y_data_emg)) / (np.max(y_data_emg) - np.min(y_data_emg))
        
        # Frekuensi sampling
        fs = 1000
        
        # Terapkan filter band-pass
        lowcut = 20.0
        highcut = 450.0
        filtered_emg = bandpass_filter(y_data_emg, lowcut, highcut, fs)

        # Hitung FFT dengan window Hamming dan padding
        windowed_data = filtered_emg * np.hamming(len(filtered_emg))   # Menerapkan window Hamming
        padded_data = np.pad(windowed_data, (0, len(filtered_emg)), 'constant')  # Padding data
        fft_result = np.fft.fft(padded_data)    # Hitung FFT
        fft_magnitude = np.abs(fft_result[:len(padded_data) // 2])   # Ambil magnitude FFT

        # Ambil frekuensi positif
        n = len(padded_data)
        frequencies = np.fft.fftfreq(len(padded_data), d=1/fs)
        positive_frequencies = frequencies[:len(padded_data) // 2]

        # Filter frekuensi di bawah 100 Hz untuk tampilan lebih jelas
        limited_frequencies = positive_frequencies[positive_frequencies < 100]
        limited_magnitude = fft_magnitude[:len(limited_frequencies)]

        # Update Grafik FFT (Grafik 2)
        self.line2.set_data(limited_frequencies, limited_magnitude)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()
        # Simpan data FFT terbaru ke self.data2 untuk penyimpanan berikutnya
        self.data2 = list(limited_magnitude)  # Menyimpan hasil FFT terbaru ke self.data2
        
        # Hitung statistik sinyal EMG
        mean_amplitude = np.mean(limited_magnitude)
        # Menghitung Median Frequency
        cumulative_sum = np.cumsum(limited_magnitude)
        half_total = 0.5 * cumulative_sum[-1]
        median_frequency = limited_frequencies[np.searchsorted(cumulative_sum, half_total)]

        
        # Tentukan status kelelahan berdasarkan mean frequency dan median frequency
        status = "Tidak Kelelahan"
        if median_frequency < 30:  # Sesuaikan nilai threshold sesuai kebutuhan
            status = "Kelelahan"
        # Update label hasil dan status
        self.mean_amplitude_label.config(text=f"Mean Amplitude: {mean_amplitude:.2f}")
        self.median_frequency_label.config(text=f"Median Frequency: {median_frequency:.2f} Hz")
        self.status_condition_label.config(text=f"Condition Status: {status}")

        print("Mean Amplitude:", mean_amplitude)
        print("Median Frequency:", median_frequency)
        

    def set_axis_range(self):
        """ Mengatur rentang sumbu pada grafik yang dipilih oleh pengguna """        
        try:
            # Mengambil nilai dari entry untuk sumbu X dan Y
            x_min, x_max = float(self.x_min_entry.get()), float(self.x_max_entry.get())
            y_min, y_max = float(self.y_min_entry.get()), float(self.y_max_entry.get())
            
           
            # Cek apakah rentang minimum lebih kecil dari maksimum
            if x_min >= x_max or y_min >= y_max:
                raise ValueError("Nilai minimum harus lebih kecil dari nilai maksimum.")
            
            # Menentukan grafik yang akan diubah rentangnya
            ax = self.ax1 if self.graph_choice.get() == "Electromyography Signal" else self.ax2
            
            # Mengatur rentang sumbu X dan Y
            ax.set_xlim(x_min, x_max)  # Mengatur rentang sumbu X
            ax.set_ylim(y_min, y_max)  # Mengatur rentang sumbu Y
            
            self.canvas.draw()  # Menggambar ulang canvas
        except ValueError:
            # Menangani error jika input tidak valid
            messagebox.showerror("Error", "Masukkan nilai numerik yang valid untuk sumbu X dan Y.")

    def save_image(self):
        """ Menyimpan gambar plot dengan status dan hasil analisis """
        try:
            # Membuat folder "data skripsi bismillah" di drive D jika belum ada
            folder_path = "record/plot"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            
            # Menyimpan gambar dengan nama file yang unik berdasarkan timestamp
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_path, f"plot_{now}.png")
            
            
            # Save plot as image
            self.fig.savefig(filename)

            # Load the saved image and add text (result_data)
            img = Image.open(filename)
            draw = ImageDraw.Draw(img)
            
            # Jika tidak ada arial.ttf, gunakan default font
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except IOError:
                font = ImageFont.load_default()  # Memuat font bawaan
            
            # Ambil teks mean amplitude, median frequency, dan status
            mean_amplitude_text = self.mean_amplitude_label.cget("text")
            median_frequency_text = self.median_frequency_label.cget("text")
            status_text = self.status_condition_label.cget("text")

            # Tambahkan teks ke dalam gambar di posisi yang berbeda
            draw.text((10, 10), mean_amplitude_text, font=font, fill="black")     # Teks mean amplitude
            draw.text((10, 30), median_frequency_text, font=font, fill="black")   # Teks median frequency
            draw.text((10, 50), status_text, font=font, fill="black")             # Teks status
            
            # Save the modified image with the text
            img.save(filename)

            # Menampilkan pesan konfirmasi
            messagebox.showinfo("Save Image", f"Image saved successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Save Image Error", f"Failed to save image: {e}")


    def save_data(self):
        """ Menyimpan data grafik EMG dan FFT dalam format CSV"""
        try:
            folder_path = "record/data"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_path, f"data_{now}.csv")
            
            
            # Konversi deque ke list agar bisa diproses
            data1_list = list(self.data1)
            data2_list = list(self.data2)
            
            
            # Buat DataFrame dari data dengan panjang berbeda
            max_length = max(len(data1_list), len(data2_list))
            
              # Padding data yang lebih pendek dengan None
            if len(data1_list) < max_length:
                data1_list.extend([None] * (max_length - len(data1_list)))
            if len(data2_list) < max_length:
                data2_list.extend([None] * (max_length - len(data2_list)))
            # Pastikan kedua data sekarang memiliki panjang yang sama
            assert len(data1_list) == len(data2_list), "Panjang data masih tidak sama setelah padding."

            # Membuat DataFrame dengan data yang sudah dipad            
            df = pd.DataFrame({
                'Time': range(max_length),
                'EMG Signal 1': data1_list,
                'EMG Signal 2': data2_list
            })

            # Jika file sudah ada, gabungkan data yang ada
            if os.path.exists(filename):
                df_existing = pd.read_csv(filename)
                 # Pastikan df_existing memiliki kolom yang sama
                if set(df_existing.columns) == set(df.columns):
                    df = pd.concat([df_existing, df], ignore_index=True)
                else:
                    print("Peringatan: Struktur kolom berbeda, membuat file baru.")

            # Simpan DataFrame ke CSV
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            messagebox.showinfo("Save Data", f"Data saved successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Save Data Error", f"Failed to save data: {e}")
            print(self.data1)
            print(self.data2)

            
                       
                
    def periodic_save(self):
        """ Menyimpan data secara berkala setiap 3 menit """
        self.save_data()  # Menyimpan data
        self.root.after(self.save_interval_ms, self.periodic_save) # Menjalankan fungsi ini kembali setelah interval tertentu

    def open_csv_file(self):
        """ Membuka file CSV dan menampilkan data pada grafik """
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.plot_excel_data(df)   # Memetakan data ke grafik
                messagebox.showinfo("Success", "CSV file successfully opened")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file\n{e}")

    
    def prompt_for_random(self):
        """ Meminta pengguna jika mereka ingin melanjutkan dengan data acak jika Arduino tidak terdeteksi """
        answer = messagebox.askyesno("Arduino Not Detected", 
                                     "Arduino not detected. Do you want to continue using random numbers?")
        if answer:
            self.use_random_data = True
        else:
            self.root.destroy()   # Menutup aplikasi jika pengguna memilih tidak melanjutkan
            
    
    def plot_excel_data(self, df):
        """ Menampilkan data yang dimuat dari file Excel pada grafik """
        self.data1.clear()  # Menghapus data sebelumnya
        self.data2.clear()
        
        # Memeriksa apakah file Excel memiliki kolom yang diperlukan
        if 'EGM Signal 1' in df.columns and 'EGM Signal 2' in df.columns:
            self.data1.extend(df['EGM Signal 1'])  # Menambahkan data kolom 'EGM Signal 1'
            self.data2.extend(df['EGM Signal 2'])  # Menambahkan data kolom 'EGM Signal 2'
        else:
            messagebox.showerror("Error", "The Excel file does not contain the required columns")
        
        # Mengatur data untuk digambar pada grafik
        self.line1.set_data(range(len(self.data1)), self.data1)
        self.line2.set_data(range(len(self.data2)), self.data2)
        
        # Menyesuaikan grafik dengan data baru
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()     # Menggambar ulang canvas
        self.canvas.flush_events()    # Membuat tampilan lebih halus
        
        # Memanggil update untuk grafik FFT setelah data dimuat
        #self.update_fft_graph()
            
    
# Bagian utama untuk menjalankan aplikasi Tkinter            
if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)  # Membuat instance dari kelas EGM_GUI
    root.mainloop()      # Menjalankan loop utama GUI
