import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import serial
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from datetime import datetime
from PIL import ImageGrab
from scipy.fft import fft
import scipy.signal as sig

cutoff_frequency = 10.0  # Frekuensi cutoff filter dalam Hz
sampling_rate = 100.0  

class EGM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoring Sinyal Elektromiografi pada Otot Manusia")
        self.root.geometry("800x600")
        
        # Setup serial communication with Arduino   
        self.serial_port = serial.Serial('COM6', 9600)  # Sesuaikan dengan port Arduino Anda
        self.navbar = tk.Menu(root)
        self.root.config(menu=self.navbar)

        # Menu "File"
        self.file_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Data", command=self.open_csv_file)  # Menghubungkan fungsi open_csv_file
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)
        
        # Menu "Options"
        self.options_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="Options", menu=self.options_menu)
        self.options_menu.add_command(label="Start", command=self.start) 
        self.options_menu.add_separator()
        self.options_menu.add_command(label="Stop", command=self.stop)
        self.options_menu.add_separator()
        self.options_menu.add_command(label="Reset", command=self.reset)

        # Menu "Save"
        self.save_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Data", command=self.save_data)
        self.save_menu.add_command(label="Save Image", command=self.save_image)

        # Menu "Analysis"
        self.analysis_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="Analysis", menu=self.analysis_menu)
        self.analysis_menu.add_command(label="Analisis FFT", command=self.calculate_fft_and_mean)

        # Footer Menu
        self.footer_frame = ttk.Frame(root)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
        self.result_label = ttk.Label(self.footer_frame, text="Nilai: N/A", anchor="e")
        self.result_label.pack(side=tk.BOTTOM, padx=550)

        self.status_label = ttk.Label(root, text="Status Kondisi: Tidak diketahui", anchor="w", font=("Helvetica", 9))
        self.status_label.pack(side=tk.BOTTOM, padx=300)

        # Create the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.ax1.set_title('Sinyal Elektromiografi')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Tegangan')
        self.line1, = self.ax1.plot([], [], lw=2)

        # Create the figure and axes for the second plot
        self.ax2.set_title('FFT')
        self.ax2.set_xlabel('Frekuensi (Hz)')
        self.ax2.set_ylabel('Amplitudo()')
        self.line2, = self.ax2.plot([], [], lw=2)

        # Create the canvas for both plots
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize data deque for both plots
        self.data1 = deque(maxlen=1000)
        self.data2 = deque(maxlen=1000)

        self.animation_running = False  # Initialize the animation_running attribute
        
        # Untuk mengubah waktu penyimpanan
        self.save_interval_ms = 3 * 60 * 1000  # Save data every 3 minutes
        self.periodic_save()

    def animation(self):
        def update_plot():
            if self.animation_running:
                # Read data from serial
                data = self.serial_port.readline().decode('ascii').strip()

                if data:
                    # Process data  
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

                self.root.after(100, update_plot)  # Call this function again after 100ms

        update_plot()  # Call the update_plot function initially

    def save_image(self):
        im = ImageGrab.grab()
        
        # Generate a unique filename using current timestamp
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"screenshot_{now}.png"
        
        # Save the image
        im.save(filename)
        print(f"Screenshot saved as {filename}")

    def save_data(self):
        # Ensure the "record" directory exists
        os.makedirs("record", exist_ok=True)

        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the filename with the timestamp
        filename = f"record/data_{timestamp}.csv"

        # Combine data1 and data2 into a DataFrame
        if len(self.data1) == len(self.data2):
            df = pd.DataFrame({
                'Time': range(len(self.data1)),
                'EGM Signal 1': list(self.data1),
                'EGM Signal 2': list(self.data2),
                'Status Kondisi': self.status_label.cget("text")
            })

            # Save the DataFrame to a CSV file
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("Error: Length of data1 and filtered_signal are not the same")

    def periodic_save(self):
        # Save data to CSV
        self.save_data()

        # Schedule the next save after the specified interval
        self.root.after(self.save_interval_ms, self.periodic_save)

    # Untuk membuka file yang sudah tersimpan   
    def open_csv_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                # Membaca file CSV
                df = pd.read_csv(file_path)
                # Menampilkan data sebagai grafik
                self.plot_excel_data(df)
                messagebox.showinfo("Success", "CSV file successfully opened")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read file\n{e}")

    def plot_excel_data(self, df):
        # Memperbarui plot dengan data dari file Excel atau CSV
        self.data1.clear()
        self.data2.clear()

        # Mengasumsikan bahwa data di kolom pertama adalah untuk plot pertama dan kolom kedua untuk plot kedua
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

    def calculate_fft_and_mean(self):
        if len(self.data1) > 0:
            # Convert deque to numpy array
            data_array = np.array(self.data1)

            # Perform FFT
            yf = fft(data_array)
            xf = np.fft.fftfreq(len(data_array), 1 / sampling_rate)

            # Calculate mean of the absolute values of FFT
            mean_fft = np.mean(np.abs(yf))

            # Display the mean FFT value
            self.result_label.config(text=f"Nilai: {mean_fft:.2f}")

            # Update status based on mean FFT value
            if mean_fft > 237:
                status = "Kelelahan"
            else:
                status = "Tidak Kelelahan"
            self.status_label.config(text=f"Status Kondisi: {status}")

            # Identify dominant frequencies
            abs_yf = np.abs(yf)
            top_indices = np.argsort(abs_yf)[-3:]  # Get indices of top 3 dominant frequencies
            top_freqs = xf[top_indices]
            top_amplitudes = abs_yf[top_indices]

            # Clear previous plot
            self.ax2.clear()
            self.ax2.plot(xf, abs_yf, label='FFT Amplitudo')
            self.ax2.scatter(top_freqs, top_amplitudes, color='red', zorder=5, label='Frekuensi Dominan')
            self.ax2.set_title('FFT')
            self.ax2.set_xlabel('Frekuensi')
            self.ax2.set_ylabel('Amplitudo')
            self.ax2.legend()

            self.canvas.draw()
            self.canvas.flush_events()

    def start(self):
        if not self.animation_running:
            self.animation_running = True
            self.animation()

    def stop(self):
        self.animation_running = False

    def reset(self):
        self.data1.clear()
        self.data2.clear()
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.canvas.draw()

    def on_closing(self):
        if self.serial_port:
            self.serial_port.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
