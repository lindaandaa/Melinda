import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EGM_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EGM Signal Viewer")

        self.navbar = tk.Menu(root)
        self.root.config(menu=self.navbar)

        self.file_menu = tk.Menu(self.navbar, tearoff=0)
        self.navbar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=root.destroy)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()

        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.status_label = ttk.Label(root, text="Status: Ready", anchor="w", font=("Helvetica", 10))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.generate_egm_data()
        self.plot_egm()

    def generate_egm_data(self):
        # Generate example EGM data
        self.time = np.linspace(0, 1, 1000)
        self.egm1 = np.sin(2 * np.pi * 5 * self.time)
        self.egm2 = np.cos(2 * np.pi * 5 * self.time)

    def plot_egm(self):
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
        

if __name__ == "__main__":
    root = tk.Tk()
    app = EGM_GUI(root)
    root.mainloop()
