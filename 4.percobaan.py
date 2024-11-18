import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
from datetime import datetime
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
lowcut = 10.0  # batas bawah frekuensi (Hz)
highcut = 500.0  # batas atas frekuensi (Hz)
fs = 1000.0  # frekuensi sampling (Hz)

# Fungsi band-pass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Frekuensi Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y




class EGM_GUI:
    def __init__(self, root):
        ### Konfigurasi dasar window  ###
        self.root = root
        self.root.title("Monitoring Electromyographic")
        self.root.geometry("1950x950")
        self.root.configure(bg='aliceblue')  #mengatur warna latar belakang
        
        
        # Frame untuk judul dan logo #
        self.title_frame = tk.Frame(root, bg="white")
        self.title_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        # Logo (Set dan ukuran)
        logo_path = "logo_kampus.png"  # Pastikan logo.png ada di folder yang sama
        self.logo_image = Image.open(logo_path)
        self.logo_image = self.logo_image.resize((70, 70), Image.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        self.logo_label = tk.Label(self.title_frame, image=self.logo_photo, bg="white")
        self.logo_label.pack(side=tk.LEFT, padx=5)
        #Judul
        self.title_label = tk.Label(self.title_frame, 
                                    text="Monitoring Electromyographic Signals in Human Muscles",
                                    font=("Helvetica", 25, "bold"), 
                                    bg="white", fg="darkblue", pady=10)
        self.title_label.pack(side=tk.TOP, padx=10, expand=True, fill=tk.Y, anchor="center")

        # Style Tombol dengan Efek Hover
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 14), padding=10, relief="solid")
        style.map("TButton",
                foreground=[('pressed', 'darkblue'), ('active', 'darkgreen')],
                background=[('pressed', '!disabled', 'lightblue'), ('active', 'lightgreen')])


        
        ### Setup Menu Bar ###
        self.navbar = tk.Menu(root,  bg="alice blue", fg="black", 
                              font=("Helvetica", 11, "bold"))
        root.config(menu=self.navbar)
        # File menu
        self.file_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=(11))
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Data", command=self.open_csv_file)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)
        # Save menu
        self.save_menu = tk.Menu(self.navbar, tearoff=0, bg="white", fg="black", font=( 11))
        self.navbar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Data", command=self.save_data)
        self.file_menu.add_separator()
        self.save_menu.add_command(label="Save Image", command=self.save_image)


        ### Options Menu ###
        self.options_menu = tk.Menu(self.navbar, tearoff=0)
        # Control Frame (Kanan)
        self.control_frame = ttk.LabelFrame(root, text="Options:")
        self.control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=True)
        # Tombol Start, Stop, Reset
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.start)
        self.start_button.pack(fill=tk.X, pady=5) 
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(fill=tk.X, pady=5)
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset)
        self.reset_button.pack(fill=tk.X, pady=5)


        ### Axis Control Frame ###
        self.axis_control_frame = ttk.LabelFrame(self.control_frame, text="Set Axis Range")
        self.axis_control_frame.pack(fill=tk.X, pady=50)
        # Pilihan grafik yang ingin diatur
        self.graph_choice_label = ttk.Label(self.axis_control_frame, text="Choose Graph:")
        self.graph_choice_label.grid(row=0, column=0, padx=5, pady=5)
        #pilihan grafik yg tersedia
        self.graph_choice = ttk.Combobox(self.axis_control_frame, values=["Electromyography Signal", "FFT"])
        self.graph_choice.grid(row=0, column=1, padx=5, pady=5)
        self.graph_choice.current(1)  # Set default pilihan ke FFT
        """Pengaturan X min, X max, Y min, Y max"""
        #Pengaturan x min
        self.x_min_label = ttk.Label(self.axis_control_frame, text="X Min:")
        self.x_min_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.x_min_entry = ttk.Entry(self.axis_control_frame)
        self.x_min_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        #Pengaturan X Max
        self.x_max_label = ttk.Label(self.axis_control_frame, text="X Max:")
        self.x_max_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.x_max_entry = ttk.Entry(self.axis_control_frame)
        self.x_max_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        #Pengaturan Y Min
        self.y_min_label = ttk.Label(self.axis_control_frame, text="Y Min:")
        self.y_min_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.y_min_entry = ttk.Entry(self.axis_control_frame)
        self.y_min_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        #Pengaturan Y Max
        self.y_max_label = ttk.Label(self.axis_control_frame, text="Y Max:")
        self.y_max_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.y_max_entry = ttk.Entry(self.axis_control_frame)
        self.y_max_entry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        # Tombol untuk mengatur sumbu sesuai dengan input
        self.set_axis_button = ttk.Button(self.axis_control_frame, text="Set Axis", command=self.set_axis_range)
        self.set_axis_button.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")
        
        
        # Analysis Frame (Analisis tombol di bawah sumbu)
        self.analysis_button = ttk.Button(self.control_frame, text="Signal Analysis", command=self.calculate_fft_and_mean)
        self.analysis_button.pack(fill=tk.X, pady=5)
        # Mengatur gaya tombol
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Helvetica", 14, "bold"), background="darkblue", foreground="black")
        
        
        ### Membuat Status dari Hasil Deteksi ###
        footer_frame = ttk.Frame(root)  # Membuat frame utama untuk footer
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        # Membuat LabelFrame untuk Status Bar "Koneksi" di sebelah kanan
        status_bar_frame = ttk.LabelFrame(footer_frame, text="Connection", borderwidth=2, relief="solid", padding=(10, 5))
        status_bar_frame.pack(side=tk.LEFT, padx=20, pady=5)
        # Label untuk menampilkan status koneksi
        status_label = ttk.Label(status_bar_frame, text="Status: Disconnected", 
                                 font=("Helvetica", 10), foreground="black", anchor="w")
        status_label.pack(fill=tk.X, padx=10)
        
        # Membuat LabelFrame untuk "Kondisi" di sebelah kiri
        condition_frame = ttk.LabelFrame(footer_frame, text="Condition", 
                                         borderwidth=2, relief="solid", padding=(10, 5))
        condition_frame.pack(side=tk.LEFT, padx=20, pady=5)
        # Label untuk menampilkan nilai kondisi dengan ukuran lebih kecil
        condition_label = ttk.Label(condition_frame, text="Value: N/A", anchor="e", 
                                    font=("Helvetica", 12), foreground="black")
        condition_label.pack(side=tk.TOP, padx=10, fill=tk.Y)
        # Label untuk menampilkan status kondisi
        status_condition_label = ttk.Label(condition_frame, text="Condition Status: Unknown", 
                                           anchor="e", font=("Helvetica", 12), foreground="black")
        status_condition_label.pack(side=tk.TOP, padx=10, fill=tk.Y)
        # Membuat LabelFrame untuk "Kondisi" di sebelah kiri
        condition_frame.pack(side=tk.LEFT, padx=100, pady=5)
        # Membuat LabelFrame untuk Status Bar "Koneksi" di sebelah kanan
        status_bar_frame.pack(side=tk.LEFT, padx=150, pady=5)
        #Misalnya, jika Anda ingin jarak lebih dekat, Anda dapat mengurangi padx menjadi 10
        
        
    
        ### Frame Untuk Grafik ###
        self.graph_frame = tk.Frame(root, bg="whitesmoke")
        self.graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=20)
        # Membuat Figure dengan ukuran lebih besar
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(20,10 ))  # Sesuaikan ukuran sesuai kebutuhan
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        # Plot pertama
        # Mengatur judul untuk ax1 dengan font yang lebih besar dan bold
        self.ax1.set_title('Electromyography Signal', fontsize=18, fontweight='bold', color="navy")
        self.ax1.set_xlabel('Time', fontsize=14)
        self.ax1.set_ylabel('Voltage', fontsize=14)
        #Ganti Background line chart 1
        self.line1, = self.ax1.plot([], [], lw=2, color='blue')
        self.ax1.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax1.legend()
        
        # Plot kedua
        # Mengatur judul untuk ax2 dengan font yang lebih besar dan bold
        self.ax2.set_title('FFT', fontsize=18, fontweight='bold', color="navy")
        self.ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        self.ax2.set_ylabel('Amplitude (A)', fontsize=14)
        #Ganti Background chart2
        self.line2, = self.ax2.plot([], [], lw=2,color='blue')
        self.ax2.set_facecolor('white')
        self.fig.patch.set_facecolor('white')
        self.ax2.legend()
        # Canvas untuk menampilkan Figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


      
        
        """Serial communication setup"""
        #Menghubungkan ke Arduino
        try:
            self.serial_port = serial.Serial('COM6', 9600)  # Sesuaikan dengan port Arduino Anda
        except:
            self.prompt_for_random()
            
        #insialisasi data dan status
        self.data1 = deque(maxlen=2000)
        self.data2 = deque(maxlen=1000)
        self.animation_running = False
        self.data_buffer = deque(maxlen=1024)  # Buffer untuk data EMG mentah




        # Automatic save interval
        self.save_interval_ms = 3 * 60 * 1000  # Save data every 3 minutes
        self.root.after(self.save_interval_ms, self.periodic_save)


    def start(self):
        """Fungsi untuk memulai animasi Pengambilan Data"""
        if not self.animation_running:
            self.animation_running = True
            self.animation()

    def stop(self):
        """"Fungsi untuk menghentikan animasi"""
        self.animation_running = False

    def reset(self):
        """ Membersihkan data, grafik, dan UI """
        self.data1.clear()
        self.data2.clear()
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        
        # Reset axis dan tampilan grafik
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()

        # Reset label dan entry
        self.result_label.config(text="Nilai: N/A")
        self.status_label.config(text="Status Kondisi: Tidak diketahui")
        for entry in [self.x_min_entry, self.x_max_entry, self.y_min_entry, self.y_max_entry]:
            entry.delete(0, tk.END)
        
    
    def animation(self):
        """ Memperbarui sinyal secara berkala jika animasi sedang berjalan."""
        def update_plot():
            if self.animation_running:  #Mengecek apakah animasi sedang berjalan
                if self.use_random_data:
                    # Menggunakan data acak jika Arduino tidak terhubung
                    value = random.uniform(0, 5)  # Generate random numbers antara 0 and 5
                else:
                    try:
                        # Kode membaca data dari Arduino
                        data = self.serial_port.readline().decode('ascii').strip()
                        if data:
                            value = float(data)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to read from Arduino: {e}")
                        return

                # Memproses sinyal
                self.data1.append(value)
                
                # Menerapkan filter Low-pass Butterworth
                if len(self.data1) > 5:  # Pastikan data cukup sebelum filter
                    b, a = sig.butter(4, 0.1, 'low')
                    filtered_signal = sig.lfilter(b, a, self.data1)
                    self.data2.append(filtered_signal[-1])
                
                # Mengupdate grafik
                self.line1.set_data(range(len(self.data1)), self.data1)
                self.ax1.relim()
                self.ax1.autoscale_view()
                self.canvas.draw()
                self.canvas.flush_events()

                # Memanggil fungsi update_plot lagi setelah 10ms
                self.root.after(10, update_plot)
        update_plot()
        
            
        
    def apply_filter(self, data, lowcut=20, highcut=450, fs=1000, order=4):
        """Fungsi untuk menerapkan bandpass filter pada sinyal."""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        filtered_data = filtfilt(b, a, data)
        return filtered_data
         
        
    def update_data(self):
        # Simulasi membaca data dari Arduino
        # Di sini, Anda bisa mengganti dengan kode membaca data dari serial Arduino
        new_data = np.random.normal(0, 1, 100)  # Ganti dengan data asli dari Arduino
        self.data1.extend(new_data)

        # Hanya menyimpan data dalam ukuran tertentu (misalnya 1000 data poin)
        if len(self.data1) > 1000:
            self.data1 = self.data1[-1000:]

        # Panggil fungsi FFT dan analisis setiap kali data diperbarui
        self.calculate_fft_and_mean()

        # Jadwalkan pemanggilan fungsi ini lagi dalam 500 ms
        self.root.after(500, self.update_data)
        
        
    def calculate_fft_and_mean(self):
        """Gabungan fungsi untuk memperbarui grafik FFT dan menghitung statistik sinyal EMG."""
        #mean dan median
        # Ambil data dari grafik EMG (data waktu)
        y_data_emg = np.array(self.line1.get_ydata())
        
        # Frekuensi sampling
        fs = 1000
        
        # Terapkan filter band-pass
        lowcut = 20.0
        highcut = 450.0
        filtered_emg = bandpass_filter(y_data_emg, lowcut, highcut, fs)

        # Hitung FFT dengan window Hamming dan padding
        windowed_data = filtered_emg * np.hamming(len(filtered_emg))
        padded_data = np.pad(windowed_data, (0, len(filtered_emg)), 'constant')
        fft_result = np.fft.fft(padded_data)
        fft_magnitude = np.abs(fft_result[:len(padded_data) // 2])

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

       
        # Hitung mean amplitude, peak frequency, mean frequency, dan median frequency
        mean_amplitude = np.mean(limited_magnitude)
        peak_frequency = limited_frequencies[np.argmax(limited_magnitude)]
        mean_frequency = np.sum(limited_frequencies * limited_magnitude) / np.sum(limited_magnitude)
        
        
        # Menghitung Median Frequency
        cumulative_sum = np.cumsum(limited_magnitude)
        half_total = 0.5 * cumulative_sum[-1]
        median_frequency = limited_frequencies[np.searchsorted(cumulative_sum, half_total)]

        
        
         # Tentukan status kelelahan berdasarkan mean frequency dan median frequency
        status = "Tidak Kelelahan"
        if mean_frequency < 30 or median_frequency < 30:  # Sesuaikan nilai threshold sesuai kebutuhan
            status = "Kelelahan"
        
        # Update label hasil dan status
        self.result_label.config(
            text=f"Mean Amplitudo: {mean_amplitude:.2f}, Mean Frequency: {mean_frequency:.2f} Hz, Median Frequency: {median_frequency:.2f} Hz"
        )
        self.status_label.config(
            text=f"Status Kondisi: {status} | Mean Amplitude: {mean_amplitude:.2f}"
        )


    def set_axis_range(self):
        """Mengatur rentang sumbu pada grafik yang dipilih oleh pengguna."""
        #graph_choice = self.graph_choice.get()
        
        try:
            x_min, x_max = float(self.x_min_entry.get()), float(self.x_max_entry.get())
            y_min, y_max = float(self.y_min_entry.get()), float(self.y_max_entry.get())
            ax = self.ax1 if self.graph_choice.get() == "Sinyal EMG" else self.ax2
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            self.canvas.draw()
        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai numerik yang valid untuk sumbu X dan Y.")

    

    def save_image(self):
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
            
            # Set font size and color (you can adjust this based on the image size)
            font = ImageFont.truetype("arial.ttf", 14)  # Use a proper font path if required

            # Define text (mean amplitude and status)
            result_text = self.result_label.cget("text")
            status_text = self.status_label.cget("text")

            # Add text to the image (at position x, y)
            draw.text((10, 10), result_text, font=font, fill="black")  # Add result text
            draw.text((10, 25), status_text, font=font, fill="black")  # Add status text

            # Save the modified image with the text
            img.save(filename)

            # Menampilkan pesan konfirmasi
            messagebox.showinfo("Save Image", f"Image saved successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Save Image Error", f"Failed to save image: {e}")


    def save_data(self):
        try:
            folder_path = "record/data"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            
            # Prompt user to choose the file location and name
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            
            if filename:
                # Get data from EMG and FFT graphs
                x_data = np.array(self.line1.get_xdata())  # Time data from EMG graph
                y_data_emg = np.array(self.line1.get_ydata())  # EMG signal data
                y_data_fft = np.array(self.line2.get_ydata())  # FFT signal data

                # Create a DataFrame for the new data
                data_to_save = pd.DataFrame({
                    'Time (s)': x_data,
                    'EMG Voltage': y_data_emg,
                    'FFT Amplitude': y_data_fft
                })
                
                # Check if the file already exists and append new data
                if os.path.exists(filename):
                    df_existing = pd.read_csv(filename)
                    # Append new data to the existing DataFrame
                    df_existing = pd.concat([df_existing, data_to_save], ignore_index=True)
                    df_existing.to_csv(filename, index=False)
                else:
                    # If the file doesn't exist, save new data
                    data_to_save.to_csv(filename, index=False)
                
                messagebox.showinfo("Success", "Data successfully saved!")
                print(f"Data saved to {filename}")
            else:
                messagebox.showwarning("File Not Chosen", "No file selected to save data.")
        except Exception as e:
            messagebox.showerror("Save Data Error", f"Failed to save data: {e}")
            print(e)


    def periodic_save(self):
        """Menyimpan data secara berkala setiap 3 menit"""
        self.save_data()
        self.root.after(self.save_interval_ms, self.periodic_save)


    def open_csv_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df = pd.read_csv(file_path)
                self.plot_excel_data(df)
                messagebox.showinfo("Success", "CSV file successfully opened")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file\n{e}")

    
    def prompt_for_random(self):
        """ Prompt user if they want to continue with random data or quit the application """
        answer = messagebox.askyesno("Arduino Not Detected", 
                                     "Arduino not detected. Do you want to continue using random numbers?")
        if answer:
            self.use_random_data = True
        else:
            self.root.destroy()
            
    
    def plot_excel_data(self, df):
        self.data1.clear()
        self.data2.clear()
        if 'EGM Signal 1' in df.columns and 'EGM Signal 2' in df.columns:
            self.data1.extend(df['EGM Signal 1'])
            self.data2.extend(df['EGM Signal 2'])
        else:
            messagebox.showerror("Error", "The Excel file does not contain the required columns")
        
        self.line1.set_data(range(len(self.data1)), self.data1)
        self.line2.set_data(range(len(self.data2)), self.data2)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()
        
         # Panggil update FFT setelah memuat data baru
        self.update_fft_graph()
            
    
       
            
       
if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)
    root.mainloop()
