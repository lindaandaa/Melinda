import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import serial

class EMG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Monitoring Sinyal Elektromiografi pada Otot Manusia")

        # Setup serial communication with Arduino
        self.serial_port = serial.Serial('COM3', 9600)  # Sesuaikan dengan port Arduino Anda

        # Main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Initialize variables for data storage
        self.max_data_points = 1000
        self.time_data = np.linspace(0, self.max_data_points-1, self.max_data_points)
        self.emg_data = np.zeros(self.max_data_points)

        # Canvas for plotting
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(self.time_data, self.emg_data, color='blue')
        self.ax.set_title('EMG Signal')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_xlim(0, self.max_data_points-1)
        self.ax.set_ylim(0, 1023)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def update_plot(self, frame):
        try:
            # Read data from Arduino
            data = self.serial_port.readline().decode().strip()
            if data:
                # Convert data to float
                value = float(data)
                # Update EMG data array
                self.emg_data = np.append(self.emg_data[1:], value)
                # Update plot
                self.line.set_ydata(self.emg_data)
                self.fig.canvas.draw()
        except Exception as e:
            print("Error while processing data:", e)

if __name__ == "__main__":
    root = tk.Tk()
    app = EMG_GUI(root)
    root.mainloop()
