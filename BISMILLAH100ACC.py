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
        self.serial_port = serial.Serial('COM6', 9600)  # Sesuaikan dengan port Arduino Anda

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

        self.data1 = deque(maxlen=2000)
        self.data2 = deque(maxlen=1000)

        self.animation_running = False

        # Automatic save interval
        self.save_interval_ms = 3 * 60 * 1000  # Save data every 3 minutes
        self.root.after(self.save_interval_ms, self.periodic_save)

    def animation(self):
        def update_plot():
            if self.animation_running:
                data = self.serial_port.readline().decode('ascii').strip()
                if data:
                    value = float(data)
                    self.data1.append(value)

                    b, a = sig.butter(4, 0.1, 'low')
                    filter_sig = sig.lfilter(b, a, self.data1)
                    self.data2.append(filter_sig[-1])

                    self.line1.set_data(range(len(self.data1)), self.data1)
                    self.ax1.relim()
                    self.ax1.autoscale_view()
                    self.canvas.draw()
                    self.canvas.flush_events()

                self.root.after(10, update_plot)

        update_plot()


    # def animation(self):
    #     def update_plot():
    #         if self.animation_running:
    #             # Mengganti data dari serial port dengan data acak
    #             value = random.uniform(0, 5)  # Menghasilkan angka acak antara 0 dan 5
    #             self.data1.append(value)

    #             b, a = sig.butter(4, 0.1, 'low')
    #             filter_sig = sig.lfilter(b, a, self.data1)
    #             self.data2.append(filter_sig[-1])

    #             self.line1.set_data(range(len(self.data1)), self.data1)
    #             self.ax1.relim()
    #             self.ax1.autoscale_view()
    #             self.canvas.draw()
    #             self.canvas.flush_events()

    #             self.root.after(10, update_plot)

    #     update_plot()

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
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(folder_path, f"data_{now}.csv")
            
            # Ensure both data1 and data2 have the same length
            min_length = min(len(self.data1), len(self.data2))
            trimmed_data1 = list(self.data1)[:min_length]
            trimmed_data2 = list(self.data2)[:min_length]

            df = pd.DataFrame({
                'Time': range(min_length),
                'EGM Signal 1': trimmed_data1,
                'EGM Signal 2': trimmed_data2
            })

            if os.path.exists(filename):
                df_existing = pd.read_csv(filename)
                df = pd.concat([df_existing, df], ignore_index=True)

            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            messagebox.showinfo("Save Data", f"Data saved successfully to {filename}")
        except Exception as e:
            messagebox.showerror("Save Data Error", f"Failed to save data: {e}")
            print(self.data1)
            print(self.data2)



    def periodic_save(self):
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

    def plot_excel_data(self, df):
        self.data1.clear()
        self.data2.clear()
        
        if 'EGM Signal 1' in df.columns and 'EGM Signal 2' in df.columns:
            self.data1.extend(df['EGM Signal 1'])
            
            # Cek apakah kolom "EGM Signal 2" mengandung NaN atau kosong
            if df['EGM Signal 2'].isnull().any():
                # Jika ada NaN, hitung FFT dan mean untuk data dari "EGM Signal 1"
                self.calculate_fft_and_mean()
            else:
                # Jika tidak ada NaN, lanjutkan plot seperti biasa
                self.data2.extend(df['EGM Signal 2'])
            
            # Set data untuk plot
            self.line1.set_data(range(len(self.data1)), self.data1)
            self.line2.set_data(range(len(self.data2)), self.data2)
            
        else:
            messagebox.showerror("Error", "The Excel file does not contain the required columns")

        # Refresh grafik
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()


    def start(self):
        if not self.animation_running:
            self.animation_running = True
            self.animation()

    def stop(self):
        self.animation_running = False

    def reset(self):
        #membersihkan data dan grafik
        self.data1.clear()
        self.data2.clear()
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        
        #membersihkan sumbu grafik
        self.ax1.autoscale()  # Mengatur ulang sumbu X dan Y secara otomatis untuk grafik EMG
        self.ax2.autoscale()  # Mengatur ulang sumbu X dan Y secara otomatis untuk grafik FFT

        #membersihkan tampilan grafik
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()
        
        #membersihkan label status dan nilai
        self.result_label.config(text="Nilai: N/A")
        self.status_label.config(text="Status Kondisi: Tidak diketahui")
        
        #membersihkan entry pada pengaturaan sumbu
        self.x_min_entry.delete(0, tk.END)
        self.x_max_entry.delete(0, tk.END)
        self.y_min_entry.delete(0, tk.END)
        self.y_max_entry.delete(0, tk.END)
        
        
    def calculate_fft_and_mean(self):
        if len(self.data1) > 0:
            signal = np.array(self.data1, dtype=float)
            fft_result = fft(signal)
            freq = np.fft.fftfreq(len(fft_result), d=1/1000)
            self.line2.set_data(freq[:len(freq)//2], np.abs(fft_result)[:len(freq)//2])
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.canvas.draw()
            self.canvas.flush_events()
            

            mean_amplitude = np.mean(np.abs(fft_result))
            self.result_label.config(text=f"Nilai: {mean_amplitude:.2f}")
            status = "Kelelahan" if mean_amplitude > 1.2 else "Tidak Kelelahan"
            self.status_label.config(text=f"Status Kondisi: {status}")
            
            # Set limits for the axes
            self.ax2.set_xlim(-10, 500)
            self.ax2.set_ylim(-10, 20000)
            
            

    def set_axis_range(self):
        graph_choice = self.graph_choice.get()
        
        try:
            x_min = float(self.x_min_entry.get())
            x_max = float(self.x_max_entry.get())
            y_min = float(self.y_min_entry.get())
            y_max = float(self.y_max_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Masukkan nilai numerik yang valid untuk sumbu X dan Y.")
            return
        
        if graph_choice == "Sinyal EMG":
            self.ax1.set_xlim(x_min, x_max)
            self.ax1.set_ylim(y_min, y_max)
        elif graph_choice == "FFT":
            self.ax2.set_xlim(x_min, x_max)
            self.ax2.set_ylim(y_min, y_max)
            self.canvas.draw()
       
if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)
    root.mainloop()
