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
import scipy.signal as sig
from scipy.interpolate import interp1d
import random
from PIL import ImageDraw, ImageFont

class EGM_GUI: #membuat inisialisasi kelas EGM_GUI
    def __init__(self, root):  #mengisialisasi elemen GUI
        self.root = root   #Menyimpan referensi ke jendela root tkinter
        self.root.title("Monitoring Electromyographic")  #mengatur judul jendela GUI
        self.root.geometry("1800x900")  #mengatur ukuran jendela GUI
        self.root.configure(bg='aliceblue')   #mengatur warna latar belkang jendela GUI

        # Menambahkan judul dan logo kampus
        self.title_frame = tk.Frame(root, bg="white") #membuat fame untuk judul di bagian atas GUI
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=5) #Menempatkan frame dibagian atas (TOP) secara horizontal (x)

        # Load dan set ukuran logo kampus
        logo_path = "logo_kampus.png"  #Path file untuk logo kampus
        self.logo_image = Image.open(logo_path)  #Membuka file gambar logo kampus menggunakan PIL
        self.logo_image = self.logo_image.resize((70, 70), Image.LANCZOS) #Mengubah ukuran gambar logo 
        self.logo_photo = ImageTk.PhotoImage(self.logo_image) #Mengonversi gambar menjadi format yang dapat digunakan oleh tkinter

        # Menampilkan logo kampus di sebelah kiri
        self.logo_label = tk.Label(self.title_frame, image=self.logo_photo, bg="white") #Membuat label tkinter untuk menampilkan logo di frame judul
        self.logo_label.pack(side=tk.LEFT, padx=5) #menempatkan label logo di sebelah kiri frame dengan padding horizontal 5 piksel

        # Menampilkan judul di tengah
        self.title_label = tk.Label(self.title_frame, text="Monitoring Electromyographic Signals in Human Muscles",
                                    font=("Helvetica", 20, "bold"), bg="white") #Membuat label tkinter untuk menampilkan judul dengan font tebal dan ukuran 20
        self.title_label.pack(side=tk.LEFT, padx=10, expand=True, fill=tk.Y) #Menempatkan label judul di sebelah kiri frame


# Serial communication setup
        try:
            self.serial_port = serial.Serial('COM6', 9600)  #mengatur komunikasi serial melalui port “com6” dengan baud rate 9600
        except:
            self.prompt_for_random()

        # Menu bar
        self.navbar = tk.Menu(root,  bg="alice blue", fg="black", font=("Helvetica", 11, "bold")) #mengatur warna latar belakang, teks dan font pada menu bar 
        root.config(menu=self.navbar)  #memastikan jendela root menampilkan menu bar

        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=(11)) #membuat menu “file” dengan latar belakang putih dan teks hitam
        self.navbar.add_cascade(label="File", menu=self.file_menu)  #menambahkan menu ke dalam navbar
        self.file_menu.add_command(label="Open Data", command=self.open_csv_file)  #menambahkan perintah “open data” untuk membuka file CSV
        self.file_menu.add_separator()  #Menambahkan garis pemisah dalam menu
        self.file_menu.add_command(label="Exit", command=root.destroy)  #menambahkan perintah "Exit" untuk menutup aplikasi
            
        # Options menu
        self.options_menu = tk.Menu(self.navbar, tearoff=0)  #membuat menu option

        # Control Frame (Kanan)
        self.control_frame = ttk.LabelFrame(root, text="Options:")  #Membuat frame label di sebelah kanan dengan judul "Options"
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)  #memposisikan frame di sisi kanan

        # Tombol Start, Stop, Reset
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start)  #Membuat tombol "Start" untuk memulai proses
        self.start_button.pack(fill=tk.X, pady=5)  #Menempatkan tombol "Start" di dalam frame
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop) #membuat tombol "Stop" untuk menghentikan proses
        self.stop_button.pack(fill=tk.X, pady=5)  #menempatkan tombol "Stop" di dalam frame
        
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)  #membuat tombol "Reset" untuk mereset proses
        self.reset_button.pack(fill=tk.X, pady=5)  #menempatkan tombol "Reset" di dalam frame

        # Save menu
        self.save_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=( 11))  #membuat menu "Save" dengan latar belakang putih dan teks hitam
        self.navbar.add_cascade(label="Save", menu=self.save_menu)  #menambahkan menu "Save" ke dalam navbar
        self.save_menu.add_command(label="Save Data", command=self.save_data)  #menambahkan perintah "Save Data" untuk menyimpan data
        self.file_menu.add_separator()   #menambahkan garis pemisah dalam menu "Save"
        self.save_menu.add_command(label="Save Image", command=self.save_image)  #menambahkan perintah "Save Image" untuk menyimpan gambar



        # Axis Control Frame
        self.axis_control_frame = ttk.LabelFrame(self.control_frame, text="Set Axis Range")  #membuat frame kontrol untuk pengaturan sumbu grafik dalam frame kontrol utama
        self.axis_control_frame.pack(fill=tk.X, pady=50)  #menempatkan frame di dalam frame kontrol

         # Pilihan grafik yang ingin diatur
        self.graph_choice_label = ttk.Label(self.axis_control_frame, text="Choose Graph:")  #membuat label untuk memilih grafik yang akan diatur
        self.graph_choice_label.grid(row=0, column=0, padx=5, pady=5)  #menempatkan label pilihan grafik di grid baris 0, kolom 0, dengan padding 5 piksel

        self.graph_choice = ttk.Combobox(self.axis_control_frame, values=["Sinyal EMG", "FFT"])  #membuat combobox untuk memilih antara grafik "Sinyal EMG" dan "FFT"
        self.graph_choice.grid(row=0, column=1, padx=5, pady=5)  #menempatkan combobox di grid baris 0, kolom 1, dengan padding 5 piksel
        self.graph_choice.current(1)   #Menyetel pilihan default combobox ke "FFT"

        # Pengaturan X min, X max, Y min, Y max
        self.x_min_label = ttk.Label(self.axis_control_frame, text="X Min:")  #membuat label untuk nilai minimum sumbu X
        self.x_min_label.grid(row=1, column=0, padx=5, pady=5)  #menempatkan label di grid baris 1, kolom 0, dengan padding 5 piksel
        self.x_min_entry = ttk.Entry(self.axis_control_frame)  #membuat entry box untuk menginput nilai minimum sumbu X
        self.x_min_entry.grid(row=1, column=1, padx=5, pady=5)  #menempatkan entry box di grid baris 1, kolom 1, dengan padding 5 piksel

        self.x_max_label = ttk.Label(self.axis_control_frame, text="X Max:")  #membuat label untuk nilai maksimum sumbu X
        self.x_max_label.grid(row=2, column=0, padx=5, pady=5)  #menempatkan label di grid baris 2, kolom 0, dengan padding 5 piksel
        self.x_max_entry = ttk.Entry(self.axis_control_frame)  #membuat entry box untuk menginput nilai maksimum sumbu X
        self.x_max_entry.grid(row=2, column=1, padx=5, pady=5)  #menempatkan entry box di grid baris 2, kolom 1, dengan padding 5 piksel

        self.y_min_label = ttk.Label(self.axis_control_frame, text="Y Min:")  #membuat label untuk nilai minimum sumbu Y
        self.y_min_label.grid(row=3, column=0, padx=5, pady=5)  #menempatkan label di grid baris 3, kolom 0, dengan padding 5 piksel
        self.y_min_entry = ttk.Entry(self.axis_control_frame)  #membuat entry box untuk menginput nilai minimum sumbu Y
        self.y_min_entry.grid(row=3, column=1, padx=5, pady=5)  #menempatkan entry box di grid baris 3, kolom 1, dengan padding 5 piksel

        self.y_max_label = ttk.Label(self.axis_control_frame, text="Y Max:")  #membuat label untuk nilai maksimum sumbu Y
        self.y_max_label.grid(row=4, column=0, padx=5, pady=5)  #menempatkan label di grid baris 4, kolom 0, dengan padding 5 piksel
        self.y_max_entry = ttk.Entry(self.axis_control_frame)  #membuat entry box untuk menginput nilai maksimum sumbu Y
        self.y_max_entry.grid(row=4, column=1, padx=5, pady=5)  #menempatkan entry box di grid baris 4, kolom 1, dengan padding 5 piksel
        # Tombol untuk mengatur sumbu sesuai dengan input
        self.set_axis_button = ttk.Button(self.axis_control_frame, text="Set Axis", command=self.set_axis_range)  #membuat tombol untuk menetapkan rentang sumbu berdasarkan input yang diberikan
        self.set_axis_button.grid(row=5, column=0, columnspan=2, pady=10)  #menempatkan tombol dan mengatur letaknya


        # Analysis Frame 
        self.analysis_frame = ttk.LabelFrame(root, text="File Analysis")   #Membuat frame untuk analisis file, ditempatkan di jendela utama (root)
        self.analysis_frame.pack(side=tk.RIGHT, pady=10)  #Menempatkan frame analisis di sisi kanan
        
        # Analysis Frame (Analisis tombol di bawah sumbu)
        self.analysis_button = ttk.Button(self.control_frame, text="Signal Analysis", command=self.calculate_fft_and_mean)  #
        self.analysis_button.pack(fill=tk.X, pady=5)  #        
        
        # Mengatur gaya tombol
        self.style = ttk.Style()  #
        self.style.configure("TButton", font=("Helvetica", 14, "bold"), background="darkgreen", foreground="black")   #

         # Footer frame
        self.footer_frame = ttk.Frame(root)  #
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, expand = True)  #



        self.status_frame = ttk.LabelFrame(self.footer_frame, text="Status", borderwidth=2, relief="groove")  #Membuat frame untuk menampilkan status di bagian bawah dengan tepi
        self.status_frame.pack(side=tk.LEFT, padx=300, pady=5)  #Menempatkan frame status di sebelah kiri 

# Label untuk menampilkan hasil nilai
        self.result_label = ttk.Label(self.status_frame, text="Nilai: N/A", anchor="e")  #membuat label untuk menampilkan nilai hasil
        self.result_label.pack(side=tk.TOP, padx=10)  #menempatkan label hasil di bagian atas frame
        
# Label untuk menampilkan status kondisi
        self.status_label = ttk.Label(self.status_frame, text="Status Kondisi: Tidak diketahui", anchor="w", font=("Helvetica", 9))  #Membuat label untuk menampilkan status kondisi
        self.status_label.pack(side=tk.TOP, padx=10)  #menempatkan label status di bagian atas frame

         # Frame untuk grafik
        self.graph_frame = tk.Frame(root, bg="mintcream")  #membuat frame untuk menampilkan grafik
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)  #menempatkan frame grafik di bagian atas
        
        # Membuat Figure dengan ukuran lebih besar
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(25,14 ))  #Membuat figure matplotlib dengan dua subplot berdampingan
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.6)  #menyesuaikan margin dan ruang antar subplots dalam figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  #menghubungkan figure matplotlib ke kanvas tkinter untuk ditampilkan dalam GUI
        
        # Plot pertama untuk sinyal EMG
        self.ax1.set_title('Electromyography Signal')  #menambahkan judul "Electromyography Signal" pada grafik pertama
        self.ax1.set_xlabel('Time')  #menambahkan label sumbu X "Time" pada grafik pertama
        self.ax1.set_ylabel('Voltage')  #menambahkan label sumbu Y "Voltage" pada grafik pertama
        self.line1, = self.ax1.plot([], [], lw=2, color='royalblue')  #menginisialisasi garis kosong untuk grafik sinyal EMG
        
        # Plot kedua untuk FFT
        self.ax2.set_title('FFT')  #menambahkan judul "FFT" pada grafik kedua
        self.ax2.set_xlabel('Frekuensi (Hz)')  #menambahkan label sumbu X "Frekuensi (Hz)" pada grafik kedua
        self.ax2.set_ylabel('Amplitudo (A)')  #menambahkan label sumbu Y "Amplitudo (A)" pada grafik kedua
        self.line2, = self.ax2.plot([], [], lw=2,color='royalblue')  #menginisialisasi garis kosong untuk grafik FFT
        
        #Mengganti warna latar belakang grafik pertama
        self.ax1.set_facecolor('white')  #mengatur latar belakang plot pertama
        self.fig.patch.set_facecolor('white')  #mengatur latar belakang keseluruhan figure 
        self.ax1.legend()  #menambahkan legenda pada grafik pertama

        #Mengganti warna latar belakang grafik kedua
        self.ax2.set_facecolor('white')  #mengatur latar belakang plot kedua 
        self.fig.patch.set_facecolor('white')  # mengatur latar belakang keseluruhan figure
        self.ax2.legend() #menambahkan legenda pada grafik kedua

        # Canvas untuk menampilkan Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)  #menghubungkan canvas untuk menampilkan figure dalam frame grafik
        self.canvas_widget = self.canvas.get_tk_widget()  #mendapatkan widget tkinter dari canvas untuk menempatkannya dalam GUI
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)  #Menempatkan widget canvas di bagian atas frame dengan pengisian penuh dan ekspansi

# Buffer untuk menyimpan data
        self.data1 = deque(maxlen=2000)  #Buffer (deque) untuk menyimpan data EMG dengan panjang maksimal 2000
        self.data2 = deque(maxlen=1000)  #Buffer (deque) untuk menyimpan data FFT dengan panjang maksimal 1000
        self.animation_running = False  #Flag untuk menunjukkan apakah animasi grafik sedang berjalan atau tidak


        # Automatic save interval
        self.save_interval_ms = 3 * 60 * 1000  # Save data every 3 minutes
        self.periodic_save()  #berfungsi untuk menyimpan data secara berkala

    def animation(self):
        def update_plot():  #mendefinisikan fungsi untuk memperbarui grafik
            if self.animation_running:  #memeriksa apakah animasi sedang berjalan
                data = self.serial_port.readline().decode('ascii').strip()  #membaca data dari port serial, mendekode menjadi string, dan menghapus spasi di awal/akhir
                if data:  #memeriksa apakah data tidak kosong
                    value = float(data)  #mengonversi data yang dibaca menjadi tipe float
                    self.data1.append(value)  #menambahkan nilai ke daftar data1

                    b, a = sig.butter(4, 0.1, 'low')  #menciptakan koefisien filter low-pass Butterworth orde 4 dengan frekuensi cutoff 0.1
                    filter_sig = sig.lfilter(b, a, self.data1)  #menerapkan filter ke data1 dan menyimpan hasilnya dalam filter_sig
                    self.data2.append(filter_sig[-1])  #menambahkan nilai terakhir dari hasil filter ke daftar data2

                    self.line1.set_data(range(len(self.data1)), self.data1)  #mengatur data grafik untuk data1
                    self.ax1.relim()  #menghitung ulang batas sumbu grafik
                    self.ax1.autoscale_view()  #mengatur ulang tampilan grafik agar sesuai dengan data baru
                    self.canvas.draw()  #menggambar ulang canvas untuk menampilkan pembaruan grafik
                    self.canvas.flush_events()  #mengosongkan event untuk memperbarui tampilan

                self.root.after(10, update_plot)  #mengatur fungsi update_plot untuk dipanggil kembali setelah 10 ms

        update_plot()  #memanggil fungsi update_plot untuk pertama kalinya


    def save_image(self):
        try:
            folder_path = "record/plot"  #menentukan path folder untuk menyimpan gambar
            if not os.path.exists(folder_path):  #memeriksa apakah folder sudah ada
                os.makedirs(folder_path, exist_ok=True)  #
            
            # Menyimpan gambar dengan nama file yang unik berdasarkan timestamp
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  #Mendapatkan waktu saat ini dalam format YYYY-MM-DD_HH-MM-SS
            filename = os.path.join(folder_path, f"plot_{now}.png")  #membuat nama file dengan timestamp
            
            # Save plot as image
            self.fig.savefig(filename)  #Menyimpan grafik sebagai gambar dengan nama file yang ditentukan

            # Load the saved image and add text (result_data)
            img = Image.open(filename)  #Membuka gambar yang baru saja disimpan
            draw = ImageDraw.Draw(img)  #Membuat objek untuk menggambar pada gambar
            
            # Set font size and color (you can adjust this based on the image size)
            font = ImageFont.truetype("arial.ttf", 14)  #Menggunakan font Arial dengan ukuran 14 (perlu path font yang tepat)

            # Define text (mean amplitude and status)
            result_text = self.result_label.cget("text") #mengambil teks hasil dari label result_label
            status_text = self.status_label.cget("text")  #mengambil teks status dari label status_label

            # Add text to the image (at position x, y)
            draw.text((10, 10), result_text, font=font, fill="black")  #menambahkan teks hasil ke gambar pada posisi (10, 10)
            draw.text((10, 25), status_text, font=font, fill="black")  #menambahkan teks status ke gambar pada posisi (10, 25)

            # Save the modified image with the text
            img.save(filename)  #Menyimpan gambar yang telah dimodifikasi (dengan teks) dengan nama file yang sama

            # Menampilkan pesan konfirmasi
            messagebox.showinfo("Save Image", f"Image saved successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Save Image Error", f"Failed to save image: {e}")

    def save_data(self):
        try:
            folder_path = "record/data"  #menentukan path folder untuk menyimpan data
            if not os.path.exists(folder_path):  #memeriksa apakah folder sudah ada
                os.makedirs(folder_path, exist_ok=True)  #jika belum ada, buat folder           
           
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  #m endapatkan waktu saat ini dalam format YYYY-MM-DD_HH-MM-SS
            filename = os.path.join(folder_path, f"data_{now}.csv")  #membuat nama file dengan timestamp
            
            # Ensure both data1 and data2 have the same length
            min_length = min(len(self.data1), len(self.data2))  # Menentukan panjang minimum antara data1 dan data2
            trimmed_data1 = list(self.data1)[:min_length]  # Memotong data1 agar memiliki panjang yang sama dengan min_length
            trimmed_data2 = list(self.data2)[:min_length]  # Memotong data2 agar memiliki panjang yang sama dengan min_length


# Membuat DataFrame dari data yang dipotong
            df = pd.DataFrame({
                'Time': range(min_length),             #membuat kolom 'Time' dari 0 hingga min_length-1
                'EGM Signal 1': trimmed_data1,    #menambahkan kolom untuk data1
                'EGM Signal 2': trimmed_data2     #menambahkan kolom untuk data2
            })

# Jika file sudah ada, baca dan gabungkan dengan data yang ada
            if os.path.exists(filename):        # Memeriksa apakah file dengan nama tersebut sudah ada
                df_existing = pd.read_csv(filename)       # Membaca data yang sudah ada dari file CSV
                df = pd.concat([df_existing, df], ignore_index=True)     # Menggabungkan data yang ada dengan data baru

            df.to_csv(filename, index=False)     #menyimpan DataFrame ke file CSV tanpa menyertakan indeks
            print(f"Data saved to {filename}")   # Mencetak pesan ke konsol yang menunjukkan bahwa data telah disimpan ke file
            messagebox.showinfo("Save Data", f"Data saved successfully to {filename}") # Menampilkan pesan sukses kepada pengguna
        except Exception as e:
            messagebox.showerror("Save Data Error", f"Failed to save data: {e}")     # Menampilkan pesan error jika terjadi kesalahan
            print(self.data1)     # Mencetak data1 ke konsol untuk debug
            print(self.data2)    # Mencetak data2 ke konsol untuk debug

    def periodic_save(self):
        self.save_data()       # Memanggil metode save_data untuk menyimpan data
        self.root.after(self.save_interval_ms, self.periodic_save)     # Mengatur panggilan kembali untuk fungsi ini setelah interval waktu yang ditentukan


    def open_csv_file(self):
        # Membuka dialog untuk memilih file CSV
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])  # Menampilkan dialog untuk memilih file CSV 
        if file_path:  # Memeriksa apakah pengguna memilih file
            try:
                #Membaca file CSV ke dalam DataFrame
                df = pd.read_csv(file_path)  #Menggunakan pandas untuk membaca file CSV ke dalam DataFrame
                self.plot_excel_data(df)    #Memanggil fungsi untuk memplot data dari DataFrame
                messagebox.showinfo("Success", "CSV file successfully opened")  #Menampilkan pesan sukses jika file berhasil dibuka
            except Exception as e:
          
      #Menangani kesalahan yang mungkin terjadi saat membaca file
                messagebox.showerror("Error", f"Failed to read file\n{e}")

    def plot_excel_data(self, df):
        self.data1.clear()  #Menghapus data sebelumnya dari data1
        self.data2.clear()  #Menghapus data sebelumnya dari data2
        #Memeriksa apakah kolom yang dibutuhkan ada dalam DataFrame
        if 'EGM Signal 1' in df.columns and 'EGM Signal 2' in df.columns:  #Memastikan kedua kolom ada dalam DataFrame
            self.data1.extend(df['EGM Signal 1'])  #Menambahkan data dari kolom 'EGM Signal 1' ke data1
            self.data2.extend(df['EGM Signal 2'])  #Menambahkan data dari kolom 'EGM Signal 2' ke data2
        else:
        #Menampilkan pesan error jika kolom tidak ditemukan
            messagebox.showerror("Error", "The Excel file does not contain the required columns")  #Menampilkan pesan error jika kolom yang dibutuhkan tidak ada

        #Memperbarui grafik dengan data baru
        self.line1.set_data(range(len(self.data1)), self.data1)  #Mengatur data untuk grafik 1
        self.line2.set_data(range(len(self.data2)), self.data2)  #Mengatur data untuk grafik 2
        self.ax1.relim()                  #Memperbarui batas sumbu untuk grafik 1
        self.ax1.autoscale_view()  #Mengatur ulang tampilan grafik 1
        self.ax2.relim()                   #Memperbarui batas sumbu untuk grafik 2
        self.ax2.autoscale_view()   #Mengatur ulang tampilan grafik 2
        self.canvas.draw()               #Menggambar ulang canvas untuk menampilkan grafik baru
        self.canvas.flush_events()   #Mengosongkan event yang tertunda dan memperbarui tampilan


    def start(self):                                     # Fungsi untuk memulai animasi
        if not self.animation_running:         #memulai animasi (mulai berjalannya sinyal)
            self.animation_running = True   #menandakan sinyal sudah berjalan
            self.animation()                          #memanggil fungsi animasi untuk memulai proses

    def stop(self):	# Fungsi untuk menghentikan animasi
        self.animation_running = False     #Menandakan bahwa animasi telah dihentikan

    def reset(self):  # Fungsi untuk mereset data dan grafik
        #membersihkan data dan grafik   
        self.data1.clear()                 #Menghapus semua data pada data1
        self.data2.clear()                #Menghapus semua data pada data2
        self.line1.set_data([], [])   #Menghapus data yang ditampilkan pada grafik 1
        self.line2.set_data([], [])   #Menghapus data yang ditampilkan pada grafik 2
        
        #membersihkan sumbu grafik
        self.ax1.autoscale()  #Mengatur ulang sumbu X dan Y secara otomatis untuk grafik EMG
        self.ax2.autoscale()  # Mengatur ulang sumbu X dan Y secara otomatis untuk grafik FFT

        #membersihkan tampilan grafik
        self.ax1.relim()                   #memperbarui batas sumbu untuk grafik EMG berdasarkan data yang ada
        self.ax1.autoscale_view()  #Mengatur ulang tampilan grafik EMG agar sesuai dengan data
        self.ax2.relim()                  #Memperbarui batas sumbu untuk grafik FFT berdasarkan data yang ada
        self.ax2.autoscale_view()  #Mengatur ulang tampilan grafik FFT agar sesuai dengan data
        self.canvas.draw()             #Menggambar ulang canvas untuk menampilkan grafik yang telah dibersihkan
        
        #membersihkan label status dan nilai
        self.result_label.config(text="Nilai: N/A")   #Mengatur label hasil menjadi "Nilai: N/A"
        self.status_label.config(text="Status Kondisi: Tidak diketahui")   #Mengatur label status menjadi "Status Kondisi: Tidak diketahui"
        
        #membersihkan entry pada pengaturaan sumbu
        self.x_min_entry.delete(0, tk.END)
        self.x_max_entry.delete(0, tk.END)   
        self.y_min_entry.delete(0, tk.END)    
        self.y_max_entry.delete(0, tk.END)


    def calculate_fft_and_mean(self):   #Fungsi untuk menghitung FFT dan mean amplitudo
        if len(self.data1) > 0:  #Memeriksa apakah data1 tidak kosong
            signal = np.array(self.data1, dtype=float)  #Mengonversi data1 menjadi array NumPy dengan tipe data float
            fft_result = fft(signal)  #Menghitung Fast Fourier Transform (FFT) dari sinyal
            freq = np.fft.fftfreq(len(fft_result), d=1/1000)  #Menghitung frekuensi bin dari hasil FFT

            #Mengatur data untuk grafik kedua (FFT)
            self.line2.set_data(freq[:len(freq)//2], np.abs(fft_result)[:len(freq)//2])  #Mengambil setengah dari frekuensi dan magnitudo FFT untuk ditampilkan
            self.ax2.relim()   #memperbarui batas sumbu untuk grafik FFT berdasarkan data yang ada
            self.ax2.autoscale_view()  #Mengatur ulang tampilan grafik FFT agar sesuai dengan data
            self.canvas.draw()             #Menggambar ulang canvas untuk menampilkan grafik yang telah diperbarui
            self.canvas.flush_events()  #Memaksa pembaruan tampilan grafik
            

            mean_amplitude = np.mean(np.abs(fft_result))   #Menghitung rata-rata amplitudo dari magnitudo FFT
            self.result_label.config(text=f"Nilai: {mean_amplitude:.2f}")    #Menampilkan nilai rata-rata amplitudo di label hasil
            status = "Kelelahan" if mean_amplitude > 1.2 else "Tidak Kelelahan"    #menentukan status berdasarkan nilai rata-rata amplitudo
            self.status_label.config(text=f"Status Kondisi: {status}")     #Menampilkan status kondisi di label status
            
            # Set limits for the axes
            self.ax2.set_xlim(-10, 1000)    #Mengatur batas sumbu X antara -10 dan 1000
            self.ax2.set_ylim(-10, 25000)  #Mengatur batas sumbu Y antara -10 dan 25000




def set_axis_range(self):   #Fungsi untuk mengatur rentang sumbu grafik berdasarkan input pengguna
        graph_choice = self.graph_choice.get()   #mengambil pilihan grafik yang dipilih oleh pengguna (Sinyal EMG atau FFT)
        
        try:
            x_min = float(self.x_min_entry.get())   #Mengambil nilai minimum sumbu X dari entry dan mengonversinya ke float
            x_max = float(self.x_max_entry.get())  #Mengambil nilai maksimum sumbu X dari entry dan mengonversinya ke float
            y_min = float(self.y_min_entry.get())   #Mengambil nilai minimum sumbu Y dari entry dan mengonversinya ke float
            y_max = float(self.y_max_entry.get())  #Mengambil nilai maksimum sumbu Y dari entry dan mengonversinya ke float
        except ValueError:   #Menangkap kesalahan jika nilai tidak bisa diubah ke float
            messagebox.showerror("Error", "Masukkan nilai numerik yang valid untuk sumbu X dan Y.")     #Menampilkan pesan kesalahan
            return   #keluar dari fungsi jika terjadi kesalahan
        
        if graph_choice == "Sinyal EMG":    #Jika pilihan adalah "Sinyal EMG"
            self.ax1.set_xlim(x_min, x_max)   #Mengatur rentang sumbu X untuk grafik sinyal EMG
            self.ax1.set_ylim(y_min, y_max)   #Mengatur rentang sumbu Y untuk grafik sinyal EMG
        elif graph_choice == "FFT":              #Jika pilihan adalah "FFT"
            self.ax2.set_xlim(x_min, x_max)  #Mengatur rentang sumbu X untuk grafik FFT
            self.ax2.set_ylim(y_min, y_max)  #Mengatur rentang sumbu Y untuk grafik FFT
            self.canvas.draw()                          #Menggambar ulang canvas untuk menampilkan perubahan rentang sumbu

       
if __name__ == "__main__":   #Memeriksa apakah file ini dieksekusi sebagai program utama
    root = tk.Tk()                        #Membuat objek Tkinter untuk antarmuka pengguna
    app = EGM_GUI(root)        #Membuat instansi dari kelas EGM_GUI
    root.mainloop()                    #Memulai loop utama Tkinter untuk menampilkan antarmuka