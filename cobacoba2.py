import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Styled GUI")

# Terapkan gaya ttk
style = ttk.Style()
style.configure("TButton", font=("Arial", 12), padding=5)  # Gaya dasar untuk tombol
style.configure("Options.TLabelframe", background="#f0f0f0")  # Background abu-abu untuk LabelFrame
style.configure("Condition.TLabelframe", background="#e0e0e0")  # Background lebih terang untuk frame "Condition"

# Fungsi untuk menambahkan ikon pada tombol (menggunakan emoji sebagai placeholder)
def set_button_icon(button, icon_text):
    button.config(text=f"{icon_text} {button['text']}")

# Frame kontrol untuk tombol dengan ikon dan efek hover
control_frame = ttk.LabelFrame(root, text="Options", style="Options.TLabelframe")
control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=True)

# Tombol Start dengan ikon
start_button = ttk.Button(control_frame, text="Start")
set_button_icon(start_button, "▶️")  # Placeholder ikon Play
start_button.pack(padx=5, pady=10, fill=tk.X)

# Tombol Stop dengan ikon
stop_button = ttk.Button(control_frame, text="Stop")
set_button_icon(stop_button, "⏹️")  # Placeholder ikon Stop
stop_button.pack(padx=5, pady=10, fill=tk.X)

# Tombol Reset dengan ikon
reset_button = ttk.Button(control_frame, text="Reset")
set_button_icon(reset_button, "🔄")  # Placeholder ikon Reset
reset_button.pack(padx=5, pady=10, fill=tk.X)

# Efek hover untuk tombol
def on_enter(e):
    e.widget['background'] = '#e6e6e6'  # Warna abu-abu terang saat hover

def on_leave(e):
    e.widget['background'] = 'SystemButtonFace'

for button in [start_button, stop_button, reset_button]:
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)

# Frame Kondisi
footer_frame = ttk.Frame(root, relief="solid", borderwidth=2)
footer_frame.place(relx=1.0, rely=1.0, anchor="se", width=290, height=130, x=-20, y=-25)

condition_frame = ttk.LabelFrame(footer_frame, text="Condition", style="Condition.TLabelframe", padding=(10, 5))
condition_frame.pack(side=tk.LEFT, padx=10, pady=5)

# Label Mean Amplitude
mean_amplitude_label = ttk.Label(condition_frame, text="Mean Amplitude: N/A", font=("Arial", 10))
mean_amplitude_label.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Label Median Frequency
median_frequency_label = ttk.Label(condition_frame, text="Median Frequency: N/A Hz", font=("Arial", 10))
median_frequency_label.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

# Label Condition Status
status_condition_label = ttk.Label(condition_frame, text="Condition Status: Unknown", font=("Arial", 10))
status_condition_label.pack(side=tk.TOP, padx=10, pady=5, fill=tk.X)

root.mainloop()
