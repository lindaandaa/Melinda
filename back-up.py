import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import serial
from collections import deque

class EGM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoring Sinyal Elektromiografi pada Otot Manusia")

        # Setup serial communication with Arduino
        self.serial_port = serial.Serial('COM3', 9600)  # Sesuaikan dengan port Arduino Anda

        

        self.animation()

    def update_plot(self):
        # Plot EGM data
        self.ax1.clear()
        self.ax1.plot(self.time, self.egm1, label='EGM 1', color='blue')
        self.ax1.set_title('EGM Signal 1')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.plot(self.time, self.egm2, label='EGM 2', color='green')
        self.ax2.set_title('EGM Signal 2')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Amplitude')
        self.ax2.legend()

        self.canvas.draw()

def read_data(self):
    try:
        while True:
            # Read data from Arduino
            data = self.serial_port.readline()
            if data:
                # Process data
                try:
                    # Lanjutkan pemrosesan data
                    # Misalnya, jika Anda ingin memplot data, lakukan di sini
                    pass
                except Exception as e:
                    print("Error while processing data:", e)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        self.serial_port.close()
 #break  # Tambahkan pernyataan break untuk keluar dari loop while



class EGM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoring Sinyal Elektromiografi pada Otot Manusia")

        # Setup serial communication with Arduino
        self.serial_port = serial.Serial('COM3', 9600)  # Sesuaikan dengan port Arduino Anda

        self.navbar = tk.Menu(root)
        self.root.config(menu=self.navbar)

        # Menu "File"
        self.file_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)

        # Menu "Options"
        self.options_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="Options", menu=self.options_menu)

        # Submenu "COM MANAGER"
        self.com_manager_menu = tk.Menu(self.options_menu, tearoff=0)
        self.options_menu.add_cascade(label="COM Manager", menu=self.com_manager_menu)
        self.com_manager_menu.add_command(label="Available Ports")
        self.com_manager_menu.add_command(label="Duration")

        self.options_menu.add_separator()

        # Submenu "Connection Manager"
        self.connection_manager_menu = tk.Menu(self.options_menu, tearoff=0)
        self.options_menu.add_cascade(label="Connection Manager", menu=self.connection_manager_menu)
        self.connection_manager_menu.add_command(label="Sync Status")
        self.connection_manager_menu.add_command(label="Active Channel")

        # Menu "Save"
        self.save_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="Save", menu=self.save_menu)
        self.save_menu.add_command(label="Save Data")
        self.save_menu.add_command(label="Save Image")

        # Footer Menu
        self.footer_frame = ttk.Frame(root)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Kotak entri untuk Gender
        self.gender_label = ttk.Label(self.footer_frame, text="Gender:", anchor="w")
        self.gender_label.pack(side=tk.LEFT)

        self.gender_entry = ttk.Entry(self.footer_frame)
        self.gender_entry.pack(side=tk.LEFT)

        # Kotak entri untuk Height
        self.height_label = ttk.Label(self.footer_frame, text="Height:", anchor="w")
        self.height_label.pack(side=tk.LEFT)

        self.height_entry = ttk.Entry(self.footer_frame)
        self.height_entry.pack(side=tk.LEFT)

        # Kotak entri untuk Age
        self.age_label = ttk.Label(self.footer_frame, text="Age:", anchor="w")
        self.age_label.pack(side=tk.LEFT)

        self.age_entry = ttk.Entry(self.footer_frame)
        self.age_entry.pack(side=tk.LEFT)

        self.result_label = ttk.Label(self.footer_frame, text="Frequency: 120 bpm\nClassification: Normal", anchor="e")
        self.result_label.pack(side=tk.RIGHT, padx=250)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.status_label = ttk.Label(root, text="Status: Ready", anchor="w", font=("Helvetica", 10))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.time = deque(maxlen=1000)  # Untuk menyimpan waktu
        self.egm1 = deque(maxlen=1000)  # Untuk menyimpan data EGM 1
        self.egm2 = deque(maxlen=1000)  # Untuk menyimpan data EGM 2



        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title('EGM Signal')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.line, = self.ax.plot([], [], lw=2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Initialize data deque
        self.data = deque(maxlen=1000)

        self.animation()

    def animation(self):
        def update_plot():
            # Read data from serial
            data = self.serial_port.readline().decode('ascii').strip()
            if data:
                # Process data
                self.data.append(float(data))
                self.line.set_data(range(len(self.data)), self.data)
                self.ax.relim()
                self.ax.autoscale_view()
                self.canvas.draw()
                self.canvas.flush_events()

            # Call this function again after 100ms
            self.root.after(100, update_plot)

        # Call the update_plot function initially
        update_plot()



# Kemudian, dalam kode yang memanggil EGM_GUI:
if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)
    app.animation()
    root.mainloop()
