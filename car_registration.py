import os
import cv2
import mysql.connector
import customtkinter as ctk
from tkinter import Toplevel, messagebox
from PIL import Image, ImageTk

# Configure CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

class CarRegistration:
    def __init__(self, root, db_connection):
        self.root = Toplevel(root)
        self.root.title("Vehicle Registration")
        self.root.geometry("500x650")
        self.root.resizable(False, False)
        self.db_connection = db_connection

        # Main Frame
        self.frame = ctk.CTkFrame(self.root, fg_color="#1E1E1E")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title Label
        self.title_label = ctk.CTkLabel(self.frame, text="Register Vehicle", font=("Arial", 18, "bold"), text_color="#FFD700")
        self.title_label.pack(pady=10)

        # Role Selection
        self.role_combobox = ctk.CTkComboBox(self.frame, values=["Student", "Staff"], fg_color="#2E2E2E", text_color="#FFD700", command=self.update_fields)
        self.role_combobox.set("Student")  # Default selection
        self.role_combobox.pack(pady=5, padx=10, fill="x")

        # Full Name Entry
        self.name_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Full Name", fg_color="#2E2E2E", text_color="#FFD700")
        self.name_entry.pack(pady=5, padx=10, fill="x")
        self.name_entry.bind("<KeyRelease>", self.convert_to_uppercase)

        # ID Entry (Matric/Staff ID)
        self.id_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Matric/Staff ID", fg_color="#2E2E2E", text_color="#FFD700")
        self.id_entry.pack(pady=5, padx=10, fill="x")

        # Plate Number Entry
        self.plate_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Plate Number", fg_color="#2E2E2E", text_color="#FFD700")
        self.plate_entry.pack(pady=5, padx=10, fill="x")
        self.plate_entry.bind("<KeyRelease>", self.convert_to_uppercase)

        # Buttons
        button_style = {"fg_color": "#FFD700", "text_color": "#000000", "hover_color": "#CDAF3E"}

        self.open_cam_btn = ctk.CTkButton(self.frame, text="Open Camera", command=self.open_camera, **button_style)
        self.open_cam_btn.pack(pady=5, padx=10, fill="x")

        self.capture_save_btn = ctk.CTkButton(self.frame, text="Capture & Save to Database", command=self.capture_and_save, **button_style)
        self.capture_save_btn.pack(pady=5, padx=10, fill="x")

        # Camera Feed Label
        self.camera_label = ctk.CTkLabel(self.frame, text="Camera Feed", height=200, fg_color="#2E2E2E", text_color="#FFD700")
        self.camera_label.pack(pady=10, padx=10, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close_window)

    def update_fields(self, choice):
        placeholder_text = "Enter Matric ID" if choice == "Student" else "Enter Staff ID"
        self.id_entry.configure(placeholder_text=placeholder_text)
        self.id_entry.delete(0, "end")

    def convert_to_uppercase(self, event):
        widget = event.widget
        current_text = widget.get()
        widget.delete(0, "end")
        widget.insert(0, current_text.upper())

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        plate_cascade = cv2.CascadeClassifier(r'C:\xampp\htdocs\unisel_security_system\haarcascade_russian_plate_number.xml')

        def update_camera_feed():
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

                for (x, y, w, h) in plates:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self.captured_plate = frame[y:y+h, x:x+w]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ctk.CTkImage(img, size=(400, 300))
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
                self.current_frame = frame

            self.root.after(10, update_camera_feed)

        update_camera_feed()


    def capture_and_save(self):
        if hasattr(self, 'current_frame'):
            self.captured_image = self.current_frame
        else:
            messagebox.showerror("Error", "Please open the camera first.")
            return

        name = self.name_entry.get()
        owner_id = self.id_entry.get()
        plate_number = self.plate_entry.get()
        role = self.role_combobox.get()

        if not name or not owner_id or not plate_number:
            messagebox.showerror("Error", "Please fill all fields and capture an image.")
            return

        _, buffer = cv2.imencode('.jpg', self.captured_image)
        image_data = buffer.tobytes()

        cursor = self.db_connection.cursor()
        try:
            cursor.execute("INSERT INTO vehicles (owner_name, matric, role, plate_number, plate_image) VALUES (%s, %s, %s, %s, %s)",
                           (name, owner_id, role, plate_number, image_data))
            self.db_connection.commit()
            messagebox.showinfo("Success", "Vehicle registered successfully!")
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error: {err}")

    def close_window(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def connect_to_database():
    try:
        db_connection = mysql.connector.connect(
            host="localhost", user="root", password="", database="unisel_security"
        )
        return db_connection
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error: {err}")
        return None

def run_app():
    root = ctk.CTk()
    root.withdraw()
    db_connection = connect_to_database()
    if db_connection:
        CarRegistration(root, db_connection)
        root.mainloop()

if __name__ == "__main__":
    run_app()
