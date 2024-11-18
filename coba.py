import serial  # untuk komunikasi serial
import time    # untuk jeda sampling
import tkinter as tk  # untuk GUI
from tkinter import messagebox, Label, Button, Frame

# KONFIGURASI KONEKSI SERIAL
PORT = 'COM3'           # Ganti dengan port yang sesuai untuk Arduino Anda
BAUDRATE = 9600         # Sesuaikan dengan baudrate Arduino
TIMEOUT = 1             # Timeout untuk serial

# KONFIGURASI THRESHOLD (BATAS) MINIMUM DAN MAKSIMUM
MIN_THRESHOLD = 100     # Batas minimum, sesuaikan sesuai kebutuhan
MAX_THRESHOLD = 800     # Batas maksimum, sesuaikan sesuai kebutuhan
FATIGUE_THRESHOLD = 450 # Threshold untuk menentukan kelelahan atau tidak kelelahan

# Inisialisasi variabel status koneksi
serial_connected = False

# FUNGSI UNTUK MEMERIKSA DAN MENGATUR KONEKSI SERIAL
def initialize_serial():
    global serial_connected, ser
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
        serial_connected = True
        connection_status.config(text="Connected", fg="green")
    except serial.SerialException:
        connection_status.config(text="Disconnected", fg="red")
        serial_connected = False

# FUNGSI UNTUK MENGECEK BATAS MINIMUM DAN MAKSIMUM
def check_threshold(value):
    if value < MIN_THRESHOLD:
        return "Below Minimum Threshold"
    elif value > MAX_THRESHOLD:
        return "Above Maximum Threshold"
    else:
        return "Normal"

# FUNGSI UNTUK MENGHITUNG RATA-RATA (MEAN) DATA
def calculate_mean(data_list):
    return sum(data_list) / len(data_list) if len(data_list) > 0 else 0

# FUNGSI UNTUK MEMPERBARUI STATUS KONDISI KELELAHAN BERDASARKAN THRESHOLD
def update_status(mean_value):
    if mean_value > FATIGUE_THRESHOLD:
        fatigue_status.config(text="Fatigued", fg="red")
    else:
        fatigue_status.config(text="Not Fatigued", fg="green")

# FUNGSI UTAMA UNTUK MENGAMBIL DAN MEMPROSES DATA
def read_serial_data():
    data_list = []
    if serial_connected:
        while serial_connected:
            if ser.in_waiting > 0:
                data = ser.readline().decode('utf-8').rstrip()
                try:
                    value = int(data)  # Konversi data menjadi integer
                    print(f"Data: {value}, Status: {check_threshold(value)}")  # Cetak nilai dan statusnya
                    data_list.append(value)

                    # Periksa rata-rata jika jumlah data mencapai 10 sampel
                    if len(data_list) >= 10:
                        mean_value = calculate_mean(data_list)
                        update_status(mean_value)
                        mean_label.config(text=f"Mean Value: {mean_value:.2f}")
                        data_list = []  # Kosongkan data_list setelah diperiksa mean-nya

                except ValueError:
                    print("Nilai tidak valid diterima.")
            time.sleep(0.1)  # Jeda sampling 100 ms

# INISIALISASI GUI UNTUK MENAMPILKAN STATUS KELELAHAN
root = tk.Tk()
root.title("EMG Monitor")
root.geometry("400x300")

# FRAME UTAMA UNTUK MENAMPILKAN STATUS KONEKSI, THRESHOLD, DAN KELELAHAN
frame = Frame(root)
frame.pack(pady=10)

# LABEL KONEKSI SERIAL
connection_status = Label(frame, text="Disconnected", font=("Arial", 12), fg="red")
connection_status.pack()

# LABEL STATUS BATAS (THRESHOLD)
threshold_status = Label(frame, text=f"Min: {MIN_THRESHOLD} | Max: {MAX_THRESHOLD}", font=("Arial", 12))
threshold_status.pack(pady=10)

# LABEL STATUS KELELAHAN
fatigue_status = Label(frame, text="Not Fatigued", font=("Arial", 14))
fatigue_status.pack(pady=10)

# LABEL UNTUK MENAMPILKAN MEAN NILAI YANG DIHITUNG
mean_label = Label(frame, text="Mean Value: -", font=("Arial", 12))
mean_label.pack(pady=10)

# BUTTON UNTUK MEMULAI PEMBACAAN DATA SERIAL
start_button = Button(root, text="Start", command=read_serial_data, font=("Arial", 12))
start_button.pack(pady=20)

# MULAI PROGRAM
initialize_serial()
if not serial_connected:
    messagebox.showerror("Connection Failed", "Make sure the Arduino device is connected to the correct port.")

root.mainloop()
