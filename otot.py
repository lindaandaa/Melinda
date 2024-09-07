import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EMG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Signal Viewer")

        # Navigasi Bar
        self.nav_frame = ttk.Frame(root)
        self.com_manager_button = ttk.Button(self.nav_frame, text="Com Manager", command=self.com_manager)
        self.connection_manager_button = ttk.Button(self.nav_frame, text="Connection Manager", command=self.connection_manager)
        self.com_manager_button.grid(row=0, column=0, padx=5, pady=5)
        self.connection_manager_button.grid(row=0, column=1, padx=5, pady=5)
        self.nav_frame.pack(side=tk.TOP, fill=tk.X)

        # Display Manager
        self.display_frame = ttk.Frame(root)
        self.display_1_label = ttk.Label(self.display_frame, text="Display 1 Manager")
        self.display_2_label = ttk.Label(self.display_frame, text="Display 2 Manager")
        self.display_1_label.grid(row=0, column=0, padx=5, pady=5)
        self.display_2_label.grid(row=0, column=1, padx=5, pady=5)
        self.display_frame.pack(side=tk.TOP, fill=tk.X)

        # Plotting Area
        self.plot_frame = ttk.Frame(root)
        self.fig, self.ax = plt.subplots(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Footer
        self.footer_frame = ttk.Frame(root)
        self.info_box = ttk.Label(self.footer_frame, text="Keterangan")
        self.result_box = ttk.Label(self.footer_frame, text="Hasil")
        self.info_box.grid(row=0, column=0, padx=5, pady=5)
        self.result_box.grid(row=0, column=1, padx=5, pady=5)
        self.footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Simulasi sinyal EMG
        self.simulate_emg()

    def simulate_emg(self):
        # Simulasi data sinyal EMG
        time = np.arange(0, 10, 0.1)
        emg_signal = np.sin(time)

        # Plot sinyal pada display 1
        self.ax.clear()
        self.ax.plot(time, emg_signal, label='EMG Signal')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('EMG Value')
        self.ax.legend()
        self.canvas.draw()

    def com_manager(self):
        # Implementasi Com Manager
        pass

    def connection_manager(self):
        # Implementasi Connection Manager
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = EMG_GUI(root)
    root.mainloop()
