import io
import os
import re
import cv2
import time
import random
import shutil
import smtplib
import easyocr
import threading
import numpy as np
import pytesseract
import tkinter as tk
import mysql.connector
import face_recognition
import customtkinter as ctk
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fpdf import FPDF
from datetime import datetime
from tkinter import filedialog
from tkcalendar import DateEntry
from db_connection import connect_db
from datetime import datetime, timedelta
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageTk, ImageDraw, ImageFont
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Toplevel, Label, Button, ttk, messagebox, Entry

# Connect to MySQL Database
db = mysql.connector.connect(host="localhost", user="root", passwd="", database="unisel_security")
cursor = db.cursor()

# Load Registered Faces (Stored as Images)
def get_registered_faces():
    cursor.execute("SELECT name, matric, role, student_image FROM students")
    students = cursor.fetchall()
    
    cursor.execute("SELECT name, matric, role, staff_image FROM staff")
    staff = cursor.fetchall()
    
    registered_faces = []
    for name, matric, role, image_data in students + staff:
        if image_data:
            nparr = np.frombuffer(image_data, np.uint8)
            stored_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            stored_encoding = face_recognition.face_encodings(stored_image)
            if stored_encoding:
                registered_faces.append((name, matric, role, stored_encoding[0], stored_image))
    return registered_faces

registered_faces = get_registered_faces()

# Load Car Plate Data
def get_registered_vehicles():
    cursor.execute("SELECT owner_name, matric, role, plate_number FROM vehicles")
    return cursor.fetchall()

registered_vehicles = get_registered_vehicles()


reader = easyocr.Reader(['en'])
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# Login Form
class LoginForm(ctk.CTkToplevel):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.title("Login")
        self.geometry("500x400")
        self.config(bg="#1E1E1E")
        
        # Center the window on the screen
        self.center_window()

        # Main Frame
        self.frame = ctk.CTkFrame(self, fg_color="#1E1E1E")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title Label
        self.title_label = ctk.CTkLabel(self.frame, text="Login", font=("Arial", 18, "bold"), text_color="#FFD700")
        self.title_label.pack(pady=20)

        # Username Entry
        self.username_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Username", fg_color="#2E2E2E", text_color="#FFD700")
        self.username_entry.pack(pady=10, padx=10, fill="x")

        # Password Entry
        self.password_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Password", fg_color="#2E2E2E", text_color="#FFD700", show="*")
        self.password_entry.pack(pady=10, padx=10, fill="x")

        # Login Button
        self.login_button = ctk.CTkButton(self.frame, text="Login", command=self.login, fg_color="#FFD700", text_color="#000000", hover_color="#CDAF3E", width=50)
        self.login_button.pack(pady=30)

        # Register Button
        self.register_button = ctk.CTkButton(self.frame, text="Don't have an account? Register", command=self.show_register_form, fg_color="#2E2E2E", text_color="#FFD700", hover_color="#CDAF3E", width=30)
        self.register_button.pack(pady=10)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # Validate user credentials (simple example)
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="unisel_security")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()

        if user:
            messagebox.showinfo("Login", "Login Successful!")
            self.destroy()  # Close the login form

            # Ensure the root window is visible and bring it to the front
            self.master.deiconify()  # Make sure the root window is visible
            self.master.lift()  # Bring the root window to the front
            self.master.attributes('-topmost', True)  # Make the root window stay on top temporarily
            self.master.attributes('-topmost', False)  # Disable topmost after bringing it to the front

            # Initialize the main system (UNISELSecuritySystem)
            app = UNISELSecuritySystem(self.master)  # Pass the same root window to the main app
            self.master.mainloop()  # Start the Tkinter main loop
        else:
            messagebox.showerror("Login", "Invalid username or password!")

        cursor.close()
        conn.close()

    def show_register_form(self):
        self.destroy()
        RegisterForm(self.master)

    def center_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        position_top = int(screen_height / 2 - 500 / 2)
        position_right = int(screen_width / 2 - 450 / 2)
        self.geometry(f'500x400+{position_right}+{position_top}')
    
# Register Form
class RegisterForm(ctk.CTkToplevel):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.title("Register")
        self.geometry("600x500")
        self.config(bg="#1E1E1E")

        # Center the window on the screen
        self.center_window()

        # Main Frame
        self.frame = ctk.CTkFrame(self, fg_color="#1E1E1E")
        self.frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Title Label
        self.title_label = ctk.CTkLabel(self.frame, text="Register", font=("Arial", 18, "bold"), text_color="#FFD700")
        self.title_label.pack(pady=20)

        # Username Entry
        self.username_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Username", fg_color="#2E2E2E", text_color="#FFD700")
        self.username_entry.pack(pady=10, padx=10, fill="x")

        # Password Entry
        self.password_entry = ctk.CTkEntry(self.frame, placeholder_text="Enter Password", fg_color="#2E2E2E", text_color="#FFD700", show="*")
        self.password_entry.pack(pady=10, padx=10, fill="x")

        # Confirm Password Entry
        self.confirm_password_entry = ctk.CTkEntry(self.frame, placeholder_text="Confirm Password", fg_color="#2E2E2E", text_color="#FFD700", show="*")
        self.confirm_password_entry.pack(pady=10, padx=10, fill="x")

        # Register Button
        self.register_button = ctk.CTkButton(self.frame, text="Register", command=self.register, fg_color="#FFD700", text_color="#000000", hover_color="#CDAF3E", width=50)
        self.register_button.pack(pady=30)
        
        # Login Button
        self.login_button = ctk.CTkButton(self.frame, text="Go to Login", command=self.login, fg_color="#FFD700", text_color="#000000", hover_color="#CDAF3E", width=50)
        self.login_button.pack(pady=30)

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        confirm_password = self.confirm_password_entry.get()

        if password != confirm_password:
            messagebox.showerror("Register", "Passwords do not match!")
            return

        conn = mysql.connector.connect(host="localhost", user="root", password="", database="unisel_security")
        cursor = conn.cursor()

        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            messagebox.showinfo("Register", "Registration Successful!")
            self.destroy()  # Close the registration window
            LoginForm(self.master)  # Open login form
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error: {err}")

        cursor.close()
        conn.close()
        
    def login(self):
        """Switch to Login Page"""
        self.destroy()  # Close the registration window
        LoginForm(self.master)  # Open the Login Form

    def center_window(self):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        position_top = int(screen_height / 2 - 450 / 2)
        position_right = int(screen_width / 2 - 400 / 2)
        self.geometry(f'600x500+{position_right}+{position_top}')

# Create Tkinter UI
class UNISELSecuritySystem:
    def __init__(self, root):
        self.root = root
        self.root.title("UNISEL SECURITY SYSTEM")
        self.root.geometry("1350x900")
        self.displayed_faces = {}
        self.displayed_plates = {}  # New: Store plates currently displayed
        self.detected_faces = set()
        self.detected_plates = set()  # New: Track plates detected but not yet displayed
        self.last_registration_time = time.time()
        self.monitoring_active = False
        
        self.db_connection = self.connect_db()
        self.reader = easyocr.Reader(['en']) 
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        self.is_running = True  # Flag to control the running status of the thread
        
        # Main Frame
        main_frame = tk.Frame(root, bg="#1E1E1E")  # Dark Grey
        main_frame.pack(fill="both", expand=True)

        # Sidebar (Navigation Panel)
        self.sidebar = tk.Frame(main_frame, width=250, bg="#181818", relief="solid")
        self.sidebar.pack(side="left", fill="y")

        # Logo / Title
        logo_label = tk.Label(self.sidebar, text="🚔 UNISEL SECURITY", font=("Arial", 16, "bold"), fg="white", bg="#181818")
        logo_label.pack(pady=20)

        # Sidebar Buttons
        self.buttons = []
        button_texts = ["📹 Monitoring", "📜 History", "🔍 Search", "🚨 Notifications", "📊 Reports", "🚪 Logout"]
        for text in button_texts:
            btn = tk.Button(self.sidebar, text=text, font=("Arial", 12), fg="white", bg="#202020",
                            relief="flat", width=20, height=2, anchor="w",
                            command=lambda t=text: self.switch_content(t))
            btn.pack(pady=5, padx=15, fill="x")
            btn.bind("<Enter>", self.on_hover)
            btn.bind("<Leave>", self.on_leave)
            self.buttons.append(btn)

        # Content Area
        self.content_frame = tk.Frame(main_frame, bg="#1E1E1E")
        self.content_frame.pack(side="right", fill="both", expand=True)
        self.label = tk.Label(self.content_frame, text="Welcome to Unisel Security System", font=("Arial", 16), fg="white", bg="#1E1E1E")
        self.label.pack(pady=20)
    
    def connect_db(self):
        """ Establishes connection to MySQL database """
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="unisel_security"
            )
            return connection
        except mysql.connector.Error as e:
            messagebox.showerror("Database Error", f"Error connecting to database: {e}")
            return None
    
  
        """Load and resize image from database and return as PhotoImage"""
        image = Image.open(io.BytesIO(image_data))  # Read from the binary stream
        image = image.resize((150, 150), Image.LANCZOS)  # Resize the image to fit the profile
        return ImageTk.PhotoImage(image)
    
    def log_access(self, identity, matric, role, entry_type):
        cursor.execute("SELECT timestamp FROM access_log WHERE identity = %s ORDER BY timestamp DESC LIMIT 1", (identity,))
        last_entry = cursor.fetchone()

        # ✅ Prevent duplicate entry (Cooldown: 10 seconds)
        if last_entry:
            last_timestamp = last_entry[0]
            if datetime.now() - last_timestamp < timedelta(seconds=10):
                return  # Skip registration if less than 10 seconds have passed

        # ✅ Insert entry into access_log
        cursor.execute("INSERT INTO access_log (identity, matric, role, entry_type, timestamp) VALUES (%s, %s, %s, %s, NOW())", 
                       (identity, matric, role, entry_type))
        db.commit()
        
    def switch_content(self, text):
        """ Update content area when switching tabs """
        # Clear previous widgets
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Remove notification references when leaving
        if text != "🚨 Notifications" and hasattr(self, "notification_tree"):
            del self.notification_tree  # Remove the Treeview to avoid reference issues

        # Handle section switching
        if text == "📹 Monitoring":
            self.init_monitoring()
        elif text == "📜 History":
            self.init_history()
        elif text == "🔍 Search":
            self.init_search()
        elif text == "🚨 Notifications":
            self.init_notifications()
        elif text == "📊 Reports":
            self.init_reports()
        elif text == "🚪 Logout":
            self.logout()
        else:
            self.label = tk.Label(self.content_frame, text=f"📌 {text}", font=("Arial", 16), fg="white", bg="#1E1E1E")
            self.label.pack(pady=20)

    def clear_content(self):
            """ Clears all content in the current content frame """
            for widget in self.content_frame.winfo_children():
                widget.destroy()

    def on_hover(self, event):
        """ Change button color on hover """
        event.widget.config(bg="#333333")

    def on_leave(self, event):
        """ Restore button color when hover ends """
        event.widget.config(bg="#202020")  

    def on_close(self):
        print("Closing the program...")
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()  # Release the camera
        self.root.quit()  # Close the Tkinter window
    
 # FACE RECOGNITION FUNCTION    
    
    def init_monitoring(self):
        """ Initialize the monitoring UI with camera feed """
        
        # ✅ Title (White Text on Dark Background)
        title_label = Label(
            self.content_frame, 
            text="📹 Monitoring", 
            font=("Arial", 16, "bold"), 
            fg="white", 
            bg="#1E1E1E"
        )
        title_label.pack(pady=10)

        # ✅ Create Two Sections: Camera (Left) & Detected Info (Right)
        frame_top = tk.Frame(self.content_frame, bg="#1E1E1E")
        frame_top.pack(fill="both", expand=True, padx=10, pady=10)

        # ✅ 📷 Live Camera Feed (Left Section)
        frame_left = tk.Frame(frame_top, bg="#181818", relief="solid", bd=2)
        frame_left.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # ✅ Camera Label (Live Feed)
        self.camera_label = Label(frame_left, bg="black", fg="white", text="🔴 Camera Not Started", font=("Arial", 12))
        self.camera_label.pack(fill="both", expand=True)

        # ✅ 🏷️ Detected Information (Right Section)
        frame_right = tk.Frame(frame_top, bg="#202020", relief="solid", bd=2)
        frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # ✅ Frame for Multiple Matched Faces (Right Panel)
        self.matched_faces_frame = tk.Frame(frame_right, bg="#202020")
        self.matched_faces_frame.pack(fill="both", expand=True, pady=10)

        # ✅ Name, Matric, Role Info
        self.matched_info = Label(frame_right, text="👤 Waiting for detection...", font=("Arial", 12), fg="white", bg="#202020")
        self.matched_info.pack(pady=10)

        # ✅ 🟢 Start Recognition Button (Visible and Centered)
        start_btn = Button(
            self.content_frame, 
            text="▶ Start Recognition", 
            command=self.start_recognition, 
            bg="#2ECC71",  # Green Color
            fg="white", 
            font=("Arial", 12, "bold"), 
            relief="flat", 
            padx=10, 
            pady=5
        )
        start_btn.pack(pady=15)

        # ✅ 🔴 Stop Monitoring Button (Visible and Centered)
        stop_btn = Button(
            self.content_frame, 
            text="⏹ Stop Monitoring", 
            command=self.stop_monitoring, 
            bg="#E74C3C",  # Red Color
            fg="white", 
            font=("Arial", 12, "bold"), 
            relief="flat", 
            padx=10, 
            pady=5
        )
        stop_btn.pack(pady=15)

    def start_recognition(self):
        """ Start the face recognition process with animated UI feedback """
        self.monitoring_active = True  # Set the flag to indicate monitoring is active
        self.cap = cv2.VideoCapture(0)  # Start capturing video from the camera
        threading.Thread(target=self.update_frame, daemon=True).start()  # Start the background thread to process frames
        
    def stop_monitoring(self):
        """ Stop the face recognition process and camera feed """
        self.monitoring_active = False  # Set the flag to indicate monitoring is stopped
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()  # Release the camera
        print("Monitoring stopped.")
        self.camera_label.config(text="🔴 Camera Stopped")  # Update UI to show that the camera has stopped

    def update_frame(self):
        """ Continuously update the live camera feed and process face and plate recognition """
        while self.monitoring_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Detect faces and process recognition
            frame, match_data_faces = self.detect_faces(frame)

            # Detect plates and process recognition
            frame, match_data_plates = self.detect_plates(frame)

            # Get the names of currently detected faces and plates
            current_detected_faces = set(match[1] for match in match_data_faces)  # Extract names from face match data
            current_detected_plates = set(match[1] for match in match_data_plates)  # Extract plate numbers from plate match data

            # Add new faces and plates to the right panel
            for match in match_data_faces:
                if match[1] not in self.displayed_faces:
                    self.display_matched_image(*match)  # Add new face

            for match in match_data_plates:
                if match[1] not in self.displayed_plates:
                    self.display_matched_plate(*match)  # Add new plate

            # Remove faces and plates that are no longer in the camera feed
            self.remove_faces_from_right_panel(current_detected_faces)
            self.remove_plates_from_right_panel(current_detected_plates)

            # Update the live feed camera in the left panel
            self.root.after(0, self.update_camera_feed, frame)

    def detect_faces(self, frame):
        """ Detect faces using face recognition """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        matched_faces = []  # To store matched face data
        recognition_threshold = 0.5  # Confidence threshold for recognition

        # Detect and track faces
        current_detected_faces = set()  # To track faces currently detected

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            best_match = None
            min_distance = float("inf")

            # Compare with registered faces in the database
            for name, matric, role, stored_encoding, stored_image in registered_faces:
                face_distances = face_recognition.face_distance([stored_encoding], face_encoding)
                if face_distances[0] < min_distance:  # Find the closest match
                    min_distance = face_distances[0]
                    best_match = (name, matric, role, stored_image)

            if best_match and min_distance < recognition_threshold:  # Only accept confident matches
                name, matric, role, stored_image = best_match
                # Draw rectangle around the face in the live feed
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Add the match data to the matched_faces list
                matched_faces.append((stored_image, name, matric, role))

                # Show the details of the matched person on screen
                text = f"{name}\n{matric}\n{role}"
                y_offset = top - 10
                for line in text.split("\n"):
                    cv2.putText(frame, line, (left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 25

                # Add to currently detected faces
                current_detected_faces.add(name)

                # Log the access after the face is recognized
                self.log_access(name, matric, role, "Face")  # Log with entry_type as "Face"

            else:
                # Handle unregistered face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "🚨 Unregistered", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Save unrecognized face and update notification
                unrecognized_face = frame[top:bottom, left:right]
                self.save_unrecognized_face(unrecognized_face)

                # Update notification
                timestamp = datetime.now()
                self.update_notifications("Face", "Unknown", unrecognized_face, timestamp)

        return frame, matched_faces if matched_faces else []  # Return matched faces data

    def save_unrecognized_face(self, face_image):
        """ Save unrecognized face to the database with rate-limiting """
        current_time = time.time()
        if current_time - self.last_registration_time < 10:
            return  # Skip registration if less than 10 seconds have passed

        try:
            timestamp = datetime.now()
            image_bytes = cv2.imencode('.jpg', face_image)[1].tobytes()

            # Insert the unrecognized face into the database
            cursor.execute("INSERT INTO unrecognized_faces (image, timestamp) VALUES (%s, %s)", (image_bytes, timestamp))
            db.commit()

            # Update the last registration time
            self.last_registration_time = current_time
            print("Unrecognized face saved successfully.")

        except Exception as e:
            print(f"Error saving unrecognized face: {e}")

    def detect_unregistered_face(self, frame, landmarks, left, top, right, bottom):
        """ Handle unregistered face using landmarks (drawing landmarks, and saving them if needed) """
        # Example: Drawing facial landmarks (e.g., eyes, nose, mouth, etc.)
        for feature in landmarks.keys():
            points = landmarks[feature]
            for point in points:
                cv2.circle(frame, point, 2, (0, 255, 255), -1)  # Yellow for landmarks

        # Optionally, log the face as unregistered in the database
        # log_unregistered_face(frame, left, top, right, bottom)

        # Display fallback message
        cv2.putText(frame, "❌ Unregistered Face Detected", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.change_ui_on_fail()

        # Here, you can choose to save the unregistered face in a separate folder or notify the admin
        # Example: cv2.imwrite("unregistered_faces/unknown_face.jpg", frame[top:bottom, left:right])

        print("Unregistered face detected with landmarks.")
    
    def update_camera_feed(self, frame=None):
        """ Update the camera feed on the monitoring screen """
        if hasattr(self, "camera_label") and self.camera_label.winfo_exists():  # Check if camera_label exists
            if frame is not None:  # Ensure frame is valid
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tk = ImageTk.PhotoImage(image=img)

                self.camera_label.imgtk = img_tk  # Retain reference to prevent garbage collection
                self.camera_label.configure(image=img_tk)
            else:
                print("⚠")
        else:
            print("⚠C")

    def update_displayed_face(self, name, img_array, matric, role):
        """ Update the image of a person already displayed in the right panel """

        if name not in self.recognized_faces:
            return  # Skip if the person is not currently displayed

        # Get the existing frame and label
        frame = self.recognized_faces[name]
        img_label = frame.winfo_children()[0]  # The image label is the first child of the frame
        details_label = frame.winfo_children()[1]  # The details label is the second child

        # Convert and resize image
        img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        img = img.resize((100, 100), Image.LANCZOS)  # Resize image to fit the panel
        img_tk = ImageTk.PhotoImage(image=img)

        # Update the image
        img_label.config(image=img_tk)
        img_label.image = img_tk  # Prevent garbage collection

        # Update the details label
        details_label.config(text=f"{name}\n{matric}\n{role}")
        
    def remove_faces_from_right_panel(self, current_detected_faces):
        """ Remove faces from the right panel if they are no longer detected """
        faces_to_remove = set(self.displayed_faces.keys()) - current_detected_faces
        for face in faces_to_remove:
            self.remove_matched_image(face)
    
    def remove_matched_image(self, name):
        """ Remove a matched image from the right panel when the person leaves the camera """
        if name in self.displayed_faces:
            frame = self.displayed_faces.pop(name)
            frame.destroy()

    def clear_right_panel(self):
        """Clear all entries from the right panel."""
        for widget in self.matched_faces_frame.winfo_children():
            widget.destroy()
        self.displayed_faces.clear()
        self.displayed_plates.clear()
        
    def display_matched_image(self, img_array, name, matric, role):
        """Display matched face images in the right panel."""
        if name in self.displayed_faces:
            return  # Skip if the face is already displayed

        try:
            # Convert the image from OpenCV (BGR) format to PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            img = img.resize((100, 100), Image.LANCZOS)  # Resize image to fit the panel
            img_tk = ImageTk.PhotoImage(image=img)

            # Add to the right panel
            total_widgets = len(self.matched_faces_frame.winfo_children())
            if total_widgets >= 25:
                self.clear_right_panel()

            row = total_widgets // 5
            column = total_widgets % 5

            frame = tk.Frame(self.matched_faces_frame, bg="#202020", relief="solid", bd=2)
            frame.grid(row=row, column=column, padx=5, pady=5)

            img_label = Label(frame, image=img_tk, bg="black")
            img_label.image = img_tk
            img_label.pack()

            details_label = Label(frame, text=f"{name}\n{matric}\n{role}", font=("Arial", 10), fg="white", bg="#202020")
            details_label.pack()

            self.displayed_faces[name] = frame  # Track the face entry

        except Exception as e:
            print(f"⚠ Error updating matched face: {e}")

# FOR PLATE DETECTION FUNCTION ONLYYY

    def detect_plates(self, frame):
        """Detect plates and match them with database records."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        plate_cascade = cv2.CascadeClassifier(r'C:\xampp\htdocs\unisel_security_system\haarcascade_russian_plate_number.xml')
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

        matches = []  # List to store detected plate matches

        for (x, y, w, h) in plates:
            plate_roi = frame[y:y + h, x:x + w]

            # Extract Plate Number
            plate_text = self.extract_plate_text(plate_roi)
            formatted_plate_text = self.format_plate_number(plate_text)

            if formatted_plate_text:
                print(f"🔍 Detected Plate: {formatted_plate_text}")

                # Match the plate number with the database
                owner_name, matric, role = self.match_plate(formatted_plate_text)

                if owner_name:
                    # Draw rectangle and display the matched details
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{owner_name} | {matric} | {role}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Add the match data for this plate
                    matches.append((plate_roi, formatted_plate_text, owner_name, matric, role))

                    # Log the plate detection
                    self.log_access(formatted_plate_text, matric, role, "Vehicle")

                else:
                    # Unregistered Plate
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "🚨 Unregistered", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Save unrecognized plate and update notification
                    unrecognized_plate = frame[y:y + h, x:x + w]
                    self.save_unrecognized_plate(unrecognized_plate, formatted_plate_text)

                    # Update notification
                    timestamp = datetime.now()
                    self.update_notifications("Vehicle", formatted_plate_text, unrecognized_plate, timestamp)

        return frame, matches

    def save_unrecognized_plate(self, plate_image, plate_number):
        """ Save unrecognized plate to the database with rate-limiting """
        current_time = time.time()
        if current_time - self.last_registration_time < 10:
            print("You can only register an unrecognized plate once every 10 seconds.")
            return  # Skip registration if less than 10 seconds have passed

        try:
            timestamp = datetime.now()
            image_bytes = cv2.imencode('.jpg', plate_image)[1].tobytes()

            # Insert the unrecognized plate into the database
            cursor = self.db_connection.cursor()
            cursor.execute("INSERT INTO unrecognized_plates (image, plate_number, timestamp) VALUES (%s, %s, %s)",
                        (image_bytes, plate_number, timestamp))
            self.db_connection.commit()

            # Update the last registration time
            self.last_registration_time = current_time
            print("Unrecognized plate saved successfully.")

        except Exception as e:
            print(f"Error saving unrecognized plate: {e}")

    def extract_plate_text(self, roi):
        text = reader.readtext(roi, detail=0)
        return text[0] if text else None
    
    def format_plate_number(self, text):
        """Clean and validate the plate number using regex"""
        pattern = r'[A-Z]{1,3}\d{1,4}[A-Z]{0,1}'
        match = re.search(pattern, text.upper().replace(" ", ""))
        if match:
            return match.group(0)
        return None
    
    def match_plate(self, plate_text):
        """Match the detected plate number with database records"""
        cursor = self.db_connection.cursor()
        cursor.execute("""
            SELECT v.plate_number, v.owner_name, v.matric, v.role
            FROM vehicles v
            WHERE v.plate_number = %s
        """, (plate_text,))

        result = cursor.fetchone()  # Fetch the first matching record
        cursor.close()

        if result:
            # Return owner_name, matric_number, and role
            return result[1], result[2], result[3]  # owner_name, matric_number, role
        else:
            return None, None, None  # If no match is found

    def display_matched_plate(self, img_array, plate_text, owner_name, matric, role):
        """Display matched plates in the right panel."""
        if plate_text in self.displayed_plates:
            return  # Skip if the plate is already displayed

        try:
            # Convert plate ROI to a displayable format
            img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            img = img.resize((100, 100), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            # Add to the right panel
            total_widgets = len(self.matched_faces_frame.winfo_children())
            if total_widgets >= 25:
                self.clear_right_panel()

            row = total_widgets // 5
            column = total_widgets % 5

            frame = tk.Frame(self.matched_faces_frame, bg="#202020", relief="solid", bd=2)
            frame.grid(row=row, column=column, padx=5, pady=5)

            img_label = Label(frame, image=img_tk, bg="black")
            img_label.image = img_tk
            img_label.pack()

            details_label = Label(frame, text=f"Plate: {plate_text}\n{owner_name}\n{matric}\n{role}", font=("Arial", 10), fg="white", bg="#202020")
            details_label.pack()

            self.displayed_plates[plate_text] = frame  # Track the plate entry

        except Exception as e:
            print(f"⚠ Error updating matched plate: {e}")
            
    def remove_plates_from_right_panel(self, current_detected_plates):
        """Remove plates from the right panel if they are no longer detected."""
        plates_to_remove = set(self.displayed_plates.keys()) - current_detected_plates
        for plate in plates_to_remove:
            self.remove_matched_plate(plate)
    
    def remove_matched_plate(self, plate_text):
        """ Remove a matched plate from the right panel when the plate leaves the camera """
        if plate_text in self.displayed_plates:
            frame = self.displayed_plates.pop(plate_text)
            frame.destroy()

# HISTORY TAB FUNCTION ONLYY

    def init_history(self):
        """ Initialize and display the access history section """
        
        # ✅ Title (White Text on Dark Background)
        title_label = tk.Label(
            self.content_frame, 
            text="📜 Access History", 
            font=("Arial", 16, "bold"), 
            fg="white", 
            bg="#1E1E1E"
        )
        title_label.pack(pady=10)

        # ✅ Create Frame for Scrollable Table (using tk.Frame with black background)
        frame = tk.Frame(self.content_frame, bg="black")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ✅ Create Treeview with columns: Name, Matric, Role, Type (Vehicle/Person), Camera, Timestamp
        self.history_tree = ttk.Treeview(frame, columns=("Name", "Matric", "Role", "Type", "Camera", "Timestamp"), show="headings")

        # Apply styling for Treeview to set black background and white font
        style = ttk.Style()
        style.configure("Treeview", background="white", foreground="black", fieldbackground="black")
        style.configure("Treeview.Heading", background="black", foreground="black")

        # Scrollbars
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=self.history_tree.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=self.history_tree.xview)
        self.history_tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        # Pack scrollbars
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")

        # Set up column headings and their width
        for col in ["Name", "Matric", "Role", "Type", "Camera", "Timestamp"]:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150, anchor="center")

        self.history_tree.pack(fill="both", expand=True)

        # ✅ "Delete All History" Button (set background and font color)
        delete_button = tk.Button(
            self.content_frame, 
            text="❌ Delete All History", 
            command=self.delete_all_history, 
            bg="red", 
            fg="white", 
            font=("Arial", 12, "bold"), 
            relief="flat", 
            padx=10, 
            pady=5
        )
        delete_button.pack(pady=10)

        # ✅ Load history data into the treeview
        self.load_history()
  
    def delete_all_history(self):
        confirm = messagebox.askyesno("⚠ Delete Confirmation", "Are you sure you want to delete all history?")
        if confirm:
            try:
                cursor.execute("DELETE FROM access_log")  # Delete all records
                db.commit()
                self.load_history()  # Refresh history table
                messagebox.showinfo("✅ Success", "All history records have been deleted.")
            except mysql.connector.Error as err:
                messagebox.showerror("❌ Error", f"Failed to delete records: {err}")

    def load_history(self):
        """ Load history data from the database and insert it into the Treeview with the newest entries at the top """
        self.history_tree.delete(*self.history_tree.get_children())  # Clear previous records

        # ✅ Load data from `access_log` with reverse order to get the most recent records first
        cursor.execute("SELECT identity, matric, role, entry_type, 'Main Gate', timestamp FROM access_log ORDER BY timestamp DESC")
        records = cursor.fetchall()

        # Insert records at the top of the Treeview
        for record in records:
            # Ensure the records are inserted starting with the most recent (based on timestamp)
            self.history_tree.insert("", "end", values=record)

# SEARCH HISTORY FUNCTION ONLY (NAME / MATRIC)

    def init_search(self):
        """ Initialize and display the search section """

        # ✅ Title (White Text on Dark Background)
        title_label = tk.Label(
            self.content_frame,
            text="🔍 Search Records",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#1E1E1E"
        )
        title_label.pack(pady=10)

        # ✅ Search Input Field
        self.search_entry = tk.Entry(
            self.content_frame,
            font=("Arial", 12),
            fg="white",
            bg="#333333",
            bd=0
        )
        self.search_entry.pack(pady=5, padx=20)
        
        # ✅ Bind Enter key to search function
        self.search_entry.bind("<Return>", lambda event: self.search_records())

        # ✅ Search Button (Rounded, Hover Effect)
        search_button = tk.Button(
            self.content_frame,
            text="Search",
            command=self.search_records,
            bg="#3498DB",
            fg="white",
            font=("Arial", 12, "bold"),
            relief="flat",
            padx=10,
            pady=5
        )
        search_button.pack(pady=5)

        def on_enter(e):
            search_button.config(bg="#2980B9")  # Darker on hover

        def on_leave(e):
            search_button.config(bg="#3498DB")  # Restore color

        search_button.bind("<Enter>", on_enter)
        search_button.bind("<Leave>", on_leave)
        
        button_frame = tk.Frame(self.content_frame, bg="#1E1E1E")
        button_frame.pack(pady=10)

        # ✅ Delete All History Button (Initially Hidden)
        self.delete_all_button = tk.Button(
            button_frame, 
            text="❌ Delete All History", 
            command=self.delete_all_records, 
            bg="darkred", 
            fg="white", 
            font=("Arial", 12, "bold"), 
            relief="flat", 
            padx=15, 
            pady=5,
            width=20  # Same width for consistency
        )
        self.delete_all_button.grid(row=0, column=1, padx=10, pady=5)  # Grid for spacing
        self.delete_all_button.grid_remove()  # Hide initially

        # ✅ Create frame for scrollable table
        frame = tk.Frame(self.content_frame, bg="black")
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ✅ Create Treeview with columns: Name, Matric, Role, Timestamp
        self.search_results = ttk.Treeview(frame, columns=("Name", "Matric", "Role", "Timestamp", "Camera"), show="headings")

        # Apply styling for Treeview
        style = ttk.Style()
        style.configure("Treeview", background="black", foreground="white", fieldbackground="black")
        style.configure("Treeview.Heading", background="black", foreground="black")

        # Scrollbars
        scroll_y = ttk.Scrollbar(frame, orient="vertical", command=self.search_results.yview)
        scroll_x = ttk.Scrollbar(frame, orient="horizontal", command=self.search_results.xview)
        self.search_results.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        # Pack scrollbars
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")

        # Set up column headings and their width
        for col in ["Name", "Matric", "Role", "Timestamp", "Camera"]:
            self.search_results.heading(col, text=col)
            self.search_results.column(col, width=150, anchor="center")

        self.search_results.pack(fill="both", expand=True)

        # ✅ Bind row selection event to show delete button
        self.search_results.bind("<ButtonRelease-1>", self.show_delete_button)

        # ✅ Delete Selected Button (Initially Hidden)
        self.delete_button = tk.Button(
            button_frame, 
            text="❌ Delete Selected", 
            command=self.delete_selected_record, 
            bg="red", 
            fg="white", 
            font=("Arial", 12, "bold"), 
            relief="flat", 
            padx=15, 
            pady=5,
            width=20  # Ensures both buttons are the same width
        )
        self.delete_button.grid(row=0, column=0, padx=10, pady=5)  # Grid for spacing
        self.delete_button.grid_remove()  # Hide initially 

    def search_records(self):
        """ Fetch search results from database """
        query = self.search_entry.get()
        cursor.execute(
            "SELECT identity, matric, role, timestamp, 'Main Gate' AS camera FROM access_log WHERE identity LIKE %s OR matric LIKE %s",
            (f"%{query}%", f"%{query}%")
        )
        records = cursor.fetchall()

        # Clear previous results
        self.search_results.delete(*self.search_results.get_children())

        # Insert new search results
        for record in records:
            self.search_results.insert("", "end", values=record)

        # Show "Delete All" button if results exist
        if records:
            self.delete_all_button.pack()
        else:
            self.delete_all_button.pack_forget()

    def show_delete_button(self, event):
        """ Show delete button when a row is selected """
        selected_item = self.search_results.selection()
        if selected_item:
            self.delete_button.pack()
        else:
            self.delete_button.pack_forget()

    def delete_selected_record(self):
        """ Delete selected record from database """
        selected_item = self.search_results.selection()
        if not selected_item:
            return

        # Get values from selected row
        record = self.search_results.item(selected_item, "values")
        identity, matric, role, timestamp = record

        # Confirm before deleting
        confirm = messagebox.askyesno("Delete Record", f"Are you sure you want to delete record for {identity}?")
        if confirm:
            cursor.execute("DELETE FROM access_log WHERE identity=%s AND matric=%s AND timestamp=%s", (identity, matric, timestamp))
            db.commit()

            # Remove from Treeview
            self.search_results.delete(selected_item)
            self.delete_button.pack_forget()

    def delete_all_records(self):
        """ Delete all searched records """
        query = self.search_entry.get()
        confirm = messagebox.askyesno("Delete All Records", f"Are you sure you want to delete all records matching '{query}'?")
        
        if confirm:
            cursor.execute("DELETE FROM access_log WHERE identity LIKE %s OR matric LIKE %s", (f"%{query}%", f"%{query}%"))
            db.commit()

            # Clear Treeview
            self.search_results.delete(*self.search_results.get_children())
            self.delete_all_button.pack_forget()
    
# NOTIFICATIONS (UNREGISTERED WARNING)    
    
    def init_notifications(self):
        """ Initialize the Notifications tab with a full-screen Treeview and right panel """
        # Clear previous content
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        # Create Main Frame
        self.notifications_frame = tk.Frame(self.content_frame, bg="#1E1E1E")
        self.notifications_frame.pack(fill="both", expand=True)

        # Left Panel - Treeview (Full Screen)
        self.tree_frame = tk.Frame(self.notifications_frame, bg="#1E1E1E")
        self.tree_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        columns = ("Type", "Timestamp", "Camera", "ID")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", selectmode="browse")

        # Configure Column Headers
        self.tree.heading("ID", text="ID")
        self.tree.heading("Type", text="Type")
        self.tree.heading("Timestamp", text="Timestamp")
        self.tree.heading("Camera", text="Camera")  # New Column

        # Adjust Column Widths
        self.tree.column("ID", width=100, anchor="center")
        self.tree.column("Type", width=200, anchor="center")
        self.tree.column("Timestamp", width=200, anchor="center")
        self.tree.column("Camera", width=150, anchor="center")  # New Column

        # Scrollbars
        tree_scroll_y = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll_y.set)
        tree_scroll_y.pack(side="right", fill="y")

        self.tree.pack(fill="both", expand=True)

        # Right Panel - Display Selected Image
        self.right_panel = tk.Frame(self.notifications_frame, width=300, bg="#202020")
        self.right_panel.pack(side="right", fill="y", padx=10, pady=10)

        self.image_label = tk.Label(self.right_panel, text="Select an image", fg="white", bg="#202020", font=("Arial", 12))
        self.image_label.pack(pady=20)

        # Button to Delete Notification
        self.delete_button = tk.Button(
            self.right_panel, text="❌ Delete Notification", command=self.delete_notification,
            bg="red", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=10, pady=5
        )
        self.delete_button.pack(pady=10)

        # Button to Delete All Notifications
        self.delete_all_button = tk.Button(
            self.right_panel, text=" Delete All Notification", command=self.delete_all_notifications,
            bg="red", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=10, pady=5
        )
        self.delete_all_button.pack(pady=10)

        # Bind Selection Event
        self.tree.bind("<<TreeviewSelect>>", self.on_item_selected)

        # Fetch and Populate Data
        self.update_treeview()

    def update_treeview(self):
        """ Update the Treeview with unrecognized records, sorted by timestamp """
        self.tree.delete(*self.tree.get_children())  # Clear old records
        combined_data = self.fetch_unrecognized_data()

        # Insert combined data (face and plate) into the Treeview
        for data in combined_data:  # Data format: (Type, Timestamp, Camera, ID, Image)
            self.tree.insert("", "end", values=(data[0], data[1], data[2], data[3]))  # Insert into Treeview

    def fetch_unrecognized_data(self):
        """ Fetch unrecognized face and plate data from the database """
        cursor = self.db_connection.cursor()

        # Query for unrecognized faces
        query_faces = "SELECT id, image, timestamp FROM unrecognized_faces ORDER BY timestamp DESC"
        cursor.execute(query_faces)
        face_data = cursor.fetchall()

        # Query for unrecognized plates
        query_plates = "SELECT id, image, plate_number, timestamp FROM unrecognized_plates ORDER BY timestamp DESC"
        cursor.execute(query_plates)
        plate_data = cursor.fetchall()

        cursor.close()

        # Combine both face and plate data into a single list and sort by timestamp
        combined_data = []
        for face in face_data:
            combined_data.append(('Unrecognized Face', face[2], "Main Gate", face[0], face[1]))  # (Type, Timestamp, Camera, ID, Image)
        for plate in plate_data:
            combined_data.append(('Unrecognized Plate', plate[3], "Main Gate", plate[0], plate[1]))  # (Type, Timestamp, Camera, ID, Image)

        return combined_data

    def on_item_selected(self, event):
        """ Display image when a row in the Treeview is selected """
        selected_item = self.tree.selection()
        if not selected_item:
            print("[INFO] No item selected.")
            return  # No selection made

        item = self.tree.item(selected_item)
        
        try:
            row_id = item["values"][3]  # Ensure ID is correctly retrieved (was [2] before)
            entry_type = item["values"][0]  # Get the Type (Face or Plate)

            cursor = self.db_connection.cursor()

            if entry_type == "Unrecognized Face":
                cursor.execute("SELECT image FROM unrecognized_faces WHERE id = %s", (row_id,))
            elif entry_type == "Unrecognized Plate":
                cursor.execute("SELECT image FROM unrecognized_plates WHERE id = %s", (row_id,))
            else:
                print("[ERROR] Invalid entry type.")
                return
            
            result = cursor.fetchone()
            cursor.close()

            if result and result[0]:  
                image_data = result[0]  # Get the image BLOB
                

                # Convert BLOB to Image
                image = Image.open(io.BytesIO(image_data))
                image = image.resize((200, 200))  # Resize for display
                image = ImageTk.PhotoImage(image)

                # Display Image in Label
                self.image_label.config(image=image, text="")
                self.image_label.image = image  # Keep a reference to prevent garbage collection

            else:
                print("[WARNING] No image found for the selected ID.")
                self.image_label.config(text="No image found", image="")

        except Exception as e:
            print("[ERROR] Fetching image failed:", e)

    def update_notifications(self, entry_type, identity, image, timestamp):
        """ Update the notification Treeview with new entries at the top """
        if not hasattr(self, "notification_tree"):
            return  # Prevents error if the tab is switched

        try:
            # Convert image to thumbnail (if required)
            thumbnail = self.create_thumbnail(image)

            # Insert new entry at the top (before the existing ones)
            self.notification_tree.insert("", 0, values=(entry_type, timestamp, thumbnail))  # '0' means top
        except Exception as e:
            print("")

    def create_thumbnail(self, image, size=(50, 50)):
        """ Create a thumbnail image for the notifications section """
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.thumbnail(size)
        img_tk = ImageTk.PhotoImage(image=image)
        return img_tk

    def delete_notification(self):
        """ Delete the selected notification from the Treeview and Database """
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("No Selection", "Please select a notification to delete.")
            return  # No selection made

        item = self.tree.item(selected_item)
        row_id = item["values"][3]  # Get the ID from the selected row

        try:
            cursor = self.db_connection.cursor()

            # Check if the ID belongs to unrecognized_faces
            cursor.execute("SELECT image FROM unrecognized_faces WHERE id = %s", (row_id,))
            result = cursor.fetchone()

            if result:
                # Delete from unrecognized_faces table
                cursor.execute("DELETE FROM unrecognized_faces WHERE id = %s", (row_id,))
            else:
                # Check if the ID belongs to unrecognized_plates
                cursor.execute("SELECT image FROM unrecognized_plates WHERE id = %s", (row_id,))
                result = cursor.fetchone()
                if result:
                    # Delete from unrecognized_plates table
                    cursor.execute("DELETE FROM unrecognized_plates WHERE id = %s", (row_id,))
                else:
                    print("No matching entry found for the selected ID.")
                    return

            # Commit the deletion to the database
            self.db_connection.commit()

            # Remove from the Treeview
            self.tree.delete(selected_item)

            # Close cursor
            cursor.close()

            # Update Treeview with the latest data
            self.update_treeview()

            # Show success message
            messagebox.showinfo("Success", "Notification deleted successfully.")

        except Exception as e:
            print("Error deleting notification:", e)
            messagebox.showerror("Error", "Failed to delete notification.")

    def delete_all_notifications(self):
        """ Delete all notifications from the Treeview and Database """
        try:
            cursor = self.db_connection.cursor()

            # Delete unrecognized faces first and commit
            cursor.execute("DELETE FROM unrecognized_faces")
            self.db_connection.commit()

            # Delete unrecognized plates next and commit
            cursor.execute("DELETE FROM unrecognized_plates")
            self.db_connection.commit()

            # Clear all records from the Treeview
            self.tree.delete(*self.tree.get_children())

            # Close cursor
            cursor.close()

            # Show success message
            messagebox.showinfo("Success", "All notifications deleted successfully.")

        except mysql.connector.Error as e:
            print("Error deleting all notifications:", e)
            messagebox.showerror("Error", "Failed to delete all notifications.")
       
# REPORT FOR AUDIT PURPOSEE

    def init_reports(self):
        """ Initializes the Reports Section with Advanced Analytics """
        # ✅ Title (Centered)
        title_label = ttk.Label(
            self.content_frame, text="📊 Reports & Advanced Analytics", 
            font=("Arial", 16, "bold"), foreground="white", background="#1E1E1E"
        )
        title_label.pack(pady=10, anchor="center")

        # ✅ Scrollable Frame for Reports
        report_frame = tk.Frame(self.content_frame, bg="#1E1E1E")
        report_frame.pack(fill="both", expand=True, padx=10, pady=10)

        canvas = tk.Canvas(report_frame, bg="#1E1E1E")
        scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1E1E1E")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="n")  # Centered
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # ✅ Display Analytics (Centered)
        self.display_event_data(scrollable_frame)
        self.display_unrecognized_chart(scrollable_frame)
        self.display_trends(scrollable_frame)
        self.display_unrecognized_pie_chart(scrollable_frame)

        # ✅ Generate Report Button (Centered)
        generate_btn = tk.Button(
            scrollable_frame, text="📄 Generate PDF Report", 
            command=self.generate_report, font=("Arial", 12, "bold"),
            bg="#2ECC71", fg="white", relief="flat", padx=10, pady=5
        )
        generate_btn.pack(pady=10)

    def display_event_data(self, parent_frame):
        """ Fetch and display grouped event data in a table """
        cursor = self.db_connection.cursor()
        query = """
            SELECT identity, role, entry_type, COUNT(*) AS frequency, 
                   MAX(timestamp) AS timestamp, 'Main Gate' AS camera
            FROM access_log
            GROUP BY identity, role, entry_type
            ORDER BY timestamp DESC
        """
        cursor.execute(query)
        data = cursor.fetchall()

        # ✅ Create Table
        columns = ("Identity", "Role", "Type", "Frequency", "Timestamp", "Camera")
        event_table = ttk.Treeview(parent_frame, columns=columns, show="headings")

        for col in columns:
            event_table.heading(col, text=col, anchor="center")
            event_table.column(col, anchor="center", width=150)

        event_table.tag_configure("dark", background="black", foreground="white")

        # ✅ Insert Data into Table
        for row in data:
            event_table.insert("", "end", values=row, tags=("dark",))

        event_table.pack(fill="both", expand=True, padx=10, pady=10)

    def fetch_unrecognized_report_data(self):
        """ Fetch Unrecognized Entry Frequency Data for Reports from DB """
        cursor = self.db_connection.cursor()
        query = """
            SELECT 'Unknown Face #' || CAST(id AS CHAR) AS identity, COUNT(*) AS frequency 
            FROM unrecognized_faces 
            GROUP BY id
            UNION ALL
            SELECT IFNULL(plate_number, 'Unknown Plate') AS identity, COUNT(*) AS frequency 
            FROM unrecognized_plates 
            WHERE plate_number IS NOT NULL
            GROUP BY plate_number
            ORDER BY frequency DESC
            LIMIT 10;
        """
        cursor.execute(query)
        return [(row[0], row[1]) for row in cursor.fetchall() if len(row) == 2]  # Ensure only two values per row

    def display_unrecognized_chart(self, parent_frame):
        """Displays a bar chart for unrecognized faces/plates with hover effects and popups."""

        # Fetch unrecognized face/plate data
        query = """
        SELECT identity, COUNT(*) as total 
        FROM (
            SELECT plate_number AS identity FROM unrecognized_plates
            UNION ALL
            SELECT 'Unregistered Face' FROM unrecognized_faces
        ) AS combined 
        GROUP BY identity 
        ORDER BY total DESC 
        LIMIT 5
        """
        cursor = self.db_connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()

        if not data:
            return  # No data available

        # Fetch images for each identity
        image_data = {}
        for identity, _ in data:
            img_query = (
                "SELECT image FROM unrecognized_faces ORDER BY timestamp DESC LIMIT 1"
                if identity == "Unregistered Face"
                else "SELECT image FROM unrecognized_plates WHERE plate_number=%s ORDER BY timestamp DESC LIMIT 1"
            )
            cursor.execute(img_query, (identity,) if identity != "Unregistered Face" else ())
            img_result = cursor.fetchone()
            if img_result:
                image_data[identity] = img_result[0]

        # Extract data
        identities = [row[0] for row in data]
        counts = [row[1] for row in data]

        # Create figure
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='black')
        ax.set_facecolor('black')

        # Bar colors
        colors = ['#e74c3c', '#f1c40f', '#3498db', '#2ECC71', '#f39c12']

        # Create bars with 3D effect
        bars = ax.bar(identities, counts, color=colors, edgecolor='black', linewidth=2, alpha=0.9)

        # Labels
        ax.set_title("Most Frequent Unrecognized Entries", fontsize=14, color="white")
        ax.set_xlabel("Identity", fontsize=12, color="white")
        ax.set_ylabel("Count", fontsize=12, color="white")
        ax.tick_params(axis='x', colors='white', rotation=20)
        ax.tick_params(axis='y', colors='white')

        # Track popup and active bar
        popup_window = None
        active_bar = None

        def show_image(event):
            """Shows image popup & applies 3D hover effect."""
            nonlocal popup_window, active_bar

            for bar, identity in zip(bars, identities):
                if bar.contains(event)[0]:  # Check if cursor is on a bar
                    if identity in image_data and image_data[identity]:
                        img_bytes = image_data[identity]
                        img = Image.open(io.BytesIO(img_bytes)).resize((150, 200))  # Resize
                        img_tk = ImageTk.PhotoImage(img)

                        x_root = canvas.get_tk_widget().winfo_pointerx()
                        y_root = canvas.get_tk_widget().winfo_pointery()

                        # If no popup exists, create it
                        if popup_window is None or not popup_window.winfo_exists():
                            popup_window = tk.Toplevel(parent_frame)
                            popup_window.overrideredirect(True)
                            popup_window.geometry(f"160x210+{x_root}+{y_root}")
                            label = tk.Label(popup_window, image=img_tk, bg="black")
                            label.image = img_tk
                            label.pack()
                        else:
                            popup_window.geometry(f"160x210+{x_root}+{y_root}")

                        # Apply 3D hover effect (increase bar size)
                        if active_bar:
                            active_bar.set_alpha(0.9)  # Reset previous bar
                            active_bar.set_linewidth(2)
                        bar.set_alpha(1.0)  # Highlight current bar
                        bar.set_linewidth(4)  # Thicker edge
                        active_bar = bar

                        canvas.draw()  # Update plot
                        return

            # If cursor leaves all bars, close popup & reset bar
            close_popup()

        def close_popup():
            """Closes image popup & resets bar style."""
            nonlocal popup_window, active_bar
            if popup_window:
                popup_window.destroy()
                popup_window = None
            if active_bar:
                active_bar.set_alpha(0.9)
                active_bar.set_linewidth(2)
                active_bar = None
                canvas.draw()

        # Hover & leave events
        fig.canvas.mpl_connect("motion_notify_event", show_image)
        fig.canvas.mpl_connect("figure_leave_event", lambda event: close_popup())

        # Embed chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

    def display_trends(self, parent_frame):
        """ Displays trends of unrecognized faces/plates with crypto-like styling and animation """
        query = """
            SELECT entry_type, COUNT(*) AS frequency, DATE(timestamp) AS date
            FROM access_log
            GROUP BY entry_type, DATE(timestamp)
            ORDER BY date DESC;
        """
        cursor.execute(query)
        data = cursor.fetchall()

        if not data:
            messagebox.showinfo("No Data", "No trends available for unrecognized events.")
            return

        dates, frequencies, entry_types = [], [], []
        for row in data:
            dates.append(row[2])  # Date
            frequencies.append(row[1])  # Frequency
            entry_types.append(row[0])  # Event Type (Face/Vehicle)

        # 📉 Trends Over Time (Animated Line Chart)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')  # Dark background
        ax.set_facecolor('black')
        
        # Use a smooth curve with a white line (crypto-like)
        ax.plot(dates, frequencies, marker='', color='white', linewidth=2)

        # Set font properties
        ax.set_xlabel("Date", fontsize=12, color="white")
        ax.set_ylabel("Frequency", fontsize=12, color="white")
        ax.set_title("Trends of Unrecognized Faces & Plates", fontsize=14, color="white")
        ax.tick_params(axis='both', labelcolor='white')

        # Smooth curve by interpolating data points
        ax.fill_between(dates, frequencies, color='white', alpha=0.1)  # Fill below the curve for better crypto-style appearance

        # Embed chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10) 

    def display_unrecognized_pie_chart(self, parent_frame):
        """Generates a pie chart for unrecognized faces and plates with hover effect"""

        # Fetch unrecognized data counts
        cursor = self.db_connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM unrecognized_faces")
        face_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM unrecognized_plates")
        plate_count = cursor.fetchone()[0]

        # Define labels and data
        labels = ["Unrecognized Faces", "Unrecognized Plates"]
        colors = ['#e74c3c', '#f1c40f']  # Red for faces, Yellow for plates
        counts = [face_count, plate_count]

        # Create figure and axes for the pie chart
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='black')

        # Create Pie Chart with 3D Effect
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            wedgeprops={'linewidth': 2, 'edgecolor': 'black', 'antialiased': True},  # 3D-like effect
            pctdistance=0.85  # Move percentages inward
        )

        # Adjust label position (shift more to the right)
        for text in texts:
            text.set_horizontalalignment('left')
            text.set_x(text.get_position()[0] + 0.2)

        # Style the chart
        ax.set_facecolor('black')
        ax.set_title("Unrecognized Faces vs. Plates", fontsize=14, color="white")

        # Change text color to white
        for text in texts + autotexts:
            text.set_color("white")

        # Hover effect: Change wedge edge thickness on hover
        def on_hover(event):
            for i, wedge in enumerate(wedges):
                if wedge.contains_point([event.x, event.y]):
                    wedge.set_linewidth(4)  # Make border thicker when hovered
                else:
                    wedge.set_linewidth(2)  # Reset thickness
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)

        # Embed pie chart in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)
   
    def generate_trends(self):
        """ Fetches trends of unrecognized faces and plates for reporting """
        query = """
            SELECT entry_type, COUNT(*) AS frequency, DATE(timestamp) AS date
            FROM access_log
            GROUP BY entry_type, DATE(timestamp)
            ORDER BY date DESC;
        """
        cursor.execute(query)
        data = cursor.fetchall()

        trends = []
        for row in data:
            trends.append({
                "entry_type": row[0],
                "frequency": row[1],
                "date": row[2]
            })
        
        return trends

    def generate_chart(data, labels, title, path, chart_type='bar'):
        plt.figure(figsize=(6, 4))
        data = [int(d) for d in data]  # Convert count values to integers

        if chart_type == 'bar':
            plt.bar(labels, data, color=['#3498db', '#2ECC71'])
        else:
            plt.pie(data, labels=labels, autopct='%1.1f%%', colors=['#3498db', '#2ECC71'])

        plt.title(title)
        plt.savefig(path)
        plt.close()

    def generate_report(self):
        """Generates a detailed security audit report with unrecognized entries and logs."""
        
        save_dir = filedialog.askdirectory(title="Choose a directory to save the report")
        if not save_dir:
            messagebox.showwarning("No Directory", "No directory selected. Report generation canceled.")
            return

        # Define report directories
        report_folder = os.path.join(os.getcwd(), "AUDIT_REPORT")
        unrecog_folder = os.path.join(report_folder, "Unrecognized_Images")
        os.makedirs(report_folder, exist_ok=True)
        os.makedirs(unrecog_folder, exist_ok=True)

        # Paths for PDF and charts
        pdf_path = os.path.join(report_folder, "Security_Audit_Report.pdf")
        chart_paths = {
            "pie_chart": os.path.join(report_folder, "pie_chart.png"),
            "bar_chart": os.path.join(report_folder, "bar_chart.png"),
            "trend_chart": os.path.join(report_folder, "trend_chart.png"),
        }

        # Database connection
        conn = mysql.connector.connect(host="localhost", user="root", password="", database="unisel_security")
        cursor = conn.cursor()

        # Fetch unrecognized data counts
        cursor.execute("SELECT COUNT(id) FROM unrecognized_faces")
        unrecognized_faces = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(id) FROM unrecognized_plates")
        unrecognized_plates = cursor.fetchone()[0]

        # Fetch top 5 frequent unrecognized plates
        cursor.execute("""
            SELECT plate_number, COUNT(*) as count FROM unrecognized_plates
            WHERE plate_number IS NOT NULL
            GROUP BY plate_number
            ORDER BY count DESC
            LIMIT 5
        """)
        top_plates = cursor.fetchall()
        top_names = [row[0] for row in top_plates]
        top_counts = [row[1] for row in top_plates]

        # Fetch trend data
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count FROM unrecognized_faces
            GROUP BY date
            ORDER BY date ASC
        """)
        trend_data = cursor.fetchall()
        dates = [row[0].strftime("%Y-%m-%d") for row in trend_data]
        date_counts = [row[1] for row in trend_data]

        # Fetch access logs
        cursor.execute("SELECT entry_type, identity, timestamp, role FROM access_log ORDER BY timestamp DESC;")
        access_log_data = cursor.fetchall()

        # Fetch unrecognized entries
        cursor.execute("SELECT id, image, timestamp FROM unrecognized_faces")
        unrecognized_faces_data = cursor.fetchall()

        cursor.execute("SELECT id, image, plate_number, timestamp FROM unrecognized_plates")
        unrecognized_plates_data = cursor.fetchall()

        # Save unrecognized images
        for face in unrecognized_faces_data:
            with open(os.path.join(unrecog_folder, f"unrecognized_face_{face[0]}.jpg"), 'wb') as f:
                f.write(face[1])

        for plate in unrecognized_plates_data:
            with open(os.path.join(unrecog_folder, f"unrecognized_plate_{plate[0]}.jpg"), 'wb') as f:
                f.write(plate[1])

        # Generate Pie Chart
        plt.figure(figsize=(5, 5))
        plt.pie([unrecognized_faces, unrecognized_plates], labels=["Unrecognized Faces", "Unrecognized Plates"], autopct='%1.1f%%')
        plt.title("Unrecognized Faces vs Plates")
        plt.savefig(chart_paths["pie_chart"])
        plt.close()

        # Generate Bar Chart
        plt.figure(figsize=(6, 4))
        plt.bar(top_names, top_counts, color='blue')
        plt.xlabel("Plate Number")
        plt.ylabel("Frequency")
        plt.title("Top 5 Unrecognized Plates")
        plt.xticks(rotation=45)
        plt.savefig(chart_paths["bar_chart"])
        plt.close()

        # Generate Trends Line Chart
        plt.figure(figsize=(6, 4))
        plt.plot(dates, date_counts, marker='o', linestyle='-', color='red')
        plt.xlabel("Date")
        plt.ylabel("Unrecognized Entries")
        plt.title("Unrecognized Trends Over Time")
        plt.xticks(rotation=45)
        plt.savefig(chart_paths["trend_chart"])
        plt.close()

        # Create PDF Report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Cover Page
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=16)
        pdf.cell(200, 10, "UNISEL SECURITY AUDIT REPORT", ln=True, align="C")
        pdf.ln(10)
        pdf.image("unisel_logo.png", x=60, w=90)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", ln=True, align="C")

        # Access Log History
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Access Log History", ln=True, align="C")
        pdf.set_font("Arial", size=10)
        pdf.cell(40, 10, "Event", border=1)
        pdf.cell(50, 10, "Identity", border=1)
        pdf.cell(50, 10, "Timestamp", border=1)
        pdf.cell(40, 10, "Role", border=1)
        pdf.ln()
        for row in access_log_data:
            pdf.cell(40, 10, row[0], border=1)
            pdf.cell(50, 10, str(row[1]), border=1)
            pdf.cell(50, 10, str(row[2]), border=1)
            pdf.cell(40, 10, str(row[3]), border=1)
            pdf.ln()

        # Unrecognized Entries
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Unrecognized Entries", ln=True, align="C")
        pdf.cell(200, 10, f"Total Unrecognized Faces: {unrecognized_faces}", ln=True)
        pdf.cell(200, 10, f"Total Unrecognized Plates: {unrecognized_plates}", ln=True)
        pdf.cell(50, 10, "ID", border=1)
        pdf.cell(70, 10, "Plate Number", border=1)
        pdf.cell(70, 10, "Timestamp", border=1)
        pdf.ln()
        for plate in unrecognized_plates_data:
            pdf.cell(50, 10, str(plate[0]), border=1)
            pdf.cell(70, 10, str(plate[2] if plate[2] else 'N/A'), border=1)
            pdf.cell(70, 10, str(plate[3]), border=1)
            pdf.ln()

        # Graphs and Trends
        pdf.add_page()
        pdf.set_font("Arial", style="B", size=14)
        pdf.cell(200, 10, "Graphs and Trends", ln=True, align="C")
        for title, path in chart_paths.items():
            pdf.ln(5)
            pdf.cell(200, 10, title.replace("_", " ").title(), ln=True, align="C")
            pdf.image(path, x=10, w=190)

        pdf.output(pdf_path)

        # Zip the AUDIT_REPORT folder
        zip_file_path = os.path.join(save_dir, "AUDIT_REPORT.zip")
        shutil.make_archive(zip_file_path.replace(".zip", ""), 'zip', report_folder)
        messagebox.showinfo("Report", f"📄 PDF Report Generated and ZIP file created! Check {zip_file_path}.")

# LOGOUT PAGE

    def logout(self):
        """ Logout functionality with confirmation dialog """
        result = messagebox.askyesno("Confirm Logout", "Are you sure you want to leave the app?")
        
        if result:  # If user clicked 'Yes'
            print("Logging out...")
            self.root.destroy()  # Forcefully destroy the Tkinter window
        else:
            print("Logout canceled.")  

if __name__ == "__main__":
    root = tk.Tk()  # Create one Tk() root window
    root.withdraw()  # Hide the root window until login is successful
    app = LoginForm(root)  # Show the login form first
    root.mainloop()  # Start the Tkinter event loop
