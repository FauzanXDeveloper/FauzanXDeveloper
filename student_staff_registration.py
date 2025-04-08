import os
import cv2
import numpy as np
import mysql.connector
import customtkinter as ctk
from tkinter import Toplevel, messagebox
from PIL import Image, ImageTk

# Configure CustomTkinter Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def create_directory(directory):
    os.makedirs(directory, exist_ok=True)

class StudentStaffRegistration:
    def __init__(self, root, db_connection):
        self.root = Toplevel(root)
        self.root.title("Student/Staff Registration")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        self.db_connection = db_connection

        # Load Face Detector and Landmark Model
        self.face_detector = cv2.CascadeClassifier(r'C:\xampp\htdocs\unisel_security_system\haarcascade_frontalface_default.xml')
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(r'C:/xampp/htdocs/unisel_security_system/fyp/lib/site-packages/cv2/data/lbfmodel.yaml')

        # Main Frame
        self.frame = ctk.CTkFrame(self.root, fg_color="#1E1E1E")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title Label
        self.title_label = ctk.CTkLabel(self.frame, text="Register Students and Staff", font=("Arial", 18, "bold"), text_color="#FFD700")
        self.title_label.pack(pady=10)

        # Role Selection
        self.role_combobox = ctk.CTkComboBox(self.frame, values=["Student", "Staff"], fg_color="#2E2E2E", text_color="#FFD700", command=self.update_fields)
        self.role_combobox.pack(pady=5, padx=10, fill="x")
        self.role_combobox.set("Student")

        # Name Entry
        self.name_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Full Name", fg_color="#2E2E2E", text_color="#FFD700")
        self.name_entry.pack(pady=5, padx=10, fill="x")
        self.name_entry.bind("<KeyRelease>", self.convert_to_uppercase)

        # Matric/Staff ID Entry
        self.matric_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Matric Number", fg_color="#2E2E2E", text_color="#FFD700")
        self.matric_entry.pack(pady=5, padx=10, fill="x")

        # Buttons
        button_style = {"fg_color": "#FFD700", "text_color": "#000000", "hover_color": "#CDAF3E"}

        self.open_cam_btn = ctk.CTkButton(self.frame, text="Open Camera", command=self.open_camera_for_registration, **button_style)
        self.open_cam_btn.pack(pady=5, padx=10, fill="x")

        self.capture_btn = ctk.CTkButton(self.frame, text="Capture & Save to Database", command=self.capture_and_save, **button_style)
        self.capture_btn.pack(pady=5, padx=10, fill="x")

        # Camera Feed Label
        self.camera_label_reg = ctk.CTkLabel(self.frame, text="Camera Feed", height=200, fg_color="#2E2E2E", text_color="#FFD700")
        self.camera_label_reg.pack(pady=10, padx=10, fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.close_window)

    def update_fields(self, choice):
        if choice == "Student":
            self.matric_entry.configure(placeholder_text="Enter Matric Number")
        else:
            self.matric_entry.configure(placeholder_text="Enter Staff ID")

    def convert_to_uppercase(self, event):
        current_text = self.name_entry.get()
        self.name_entry.delete(0, "end")
        self.name_entry.insert(0, current_text.upper())

    def open_camera_for_registration(self):
        self.cap = cv2.VideoCapture(0)

        def update_camera_feed():
            ret, frame = self.cap.read()
            if ret:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_detector.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                for (x, y, w, h) in faces:
                    face_roi = frame_gray[y:y+h, x:x+w]
                    landmarks = self.detect_landmarks(face_roi, x, y)

                    for landmark in landmarks:
                        for i, (px, py) in enumerate(landmark):
                            # Exclude eye landmarks (indexes 36-41 and 42-47)
                            if 36 <= i <= 41 or 42 <= i <= 47:
                                continue
                            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(img)
                self.camera_label_reg.imgtk = imgtk
                self.camera_label_reg.configure(image=imgtk)
                self.current_frame = frame

            self.root.after(10, update_camera_feed)

        update_camera_feed()

    def detect_landmarks(self, face_roi, offset_x, offset_y):
        landmarks_list = []
        _, landmarks = self.landmark_detector.fit(face_roi, np.array([[0, 0, face_roi.shape[1], face_roi.shape[0]]]))
        if landmarks:
            for landmark in landmarks:
                landmarks_list.append([(int(x + offset_x), int(y + offset_y)) for x, y in landmark[0]])
        return landmarks_list

    def capture_and_save(self):
        if hasattr(self, 'current_frame'):
            self.captured_image = self.current_frame
        else:
            messagebox.showerror("Error", "Please open the camera first.")
            return

        name = self.name_entry.get()
        matric = self.matric_entry.get()
        role = self.role_combobox.get()

        if not name or not matric or not role:
            messagebox.showerror("Error", "Please fill all fields and capture an image.")
            return

        _, buffer = cv2.imencode('.jpg', self.captured_image)
        image_data = buffer.tobytes()

        cursor = self.db_connection.cursor()
        table_name = "students" if role == "Student" else "staff"
        image_column = "student_image" if role == "Student" else "staff_image"

        try:
            cursor.execute(f"INSERT INTO {table_name} (name, matric, role, {image_column}) VALUES (%s, %s, %s, %s)",
                           (name, matric, role, image_data))
            self.db_connection.commit()
            messagebox.showinfo("Success", "Record saved successfully!")
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error: {err}")

    def close_window(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

def connect_to_database():
    try:
        db_connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="unisel_security"
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
        app = StudentStaffRegistration(root, db_connection)
        root.mainloop()

if __name__ == "__main__":
    run_app()
