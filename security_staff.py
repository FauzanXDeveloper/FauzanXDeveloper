import customtkinter as ctk
import cv2
import re
import io
import numpy as np
import face_recognition
import easyocr
import threading
import tkinter as tk
import time
from tkinter import messagebox
from PIL import Image, ImageTk
import mysql.connector
import pytesseract
from tkinter import Toplevel, Label, Button, ttk, messagebox, Entry
from datetime import datetime, timedelta

# Database connection setup
db = mysql.connector.connect(host="localhost", user="root", passwd="", database="unisel_security")
cursor = db.cursor()

# EasyOCR setup for vehicle plates
reader = easyocr.Reader(['en'])

# Function to get registered faces and vehicles from the database
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

# Function to get registered vehicles
def get_registered_vehicles():
    cursor.execute("SELECT owner_name, matric, role, plate_number FROM vehicles")
    return cursor.fetchall()

registered_vehicles = get_registered_vehicles()

# Security Staff UI
class SecurityStaff:
    def __init__(self, root):
        self.root = root
        self.root.title("UNISEL SECURITY STAFF")
        self.root.geometry("1350x900")

        self.displayed_faces = {}
        self.displayed_plates = {}
        self.detected_faces = set()
        self.detected_plates = set()

        self.db_connection = db
        self.reader = easyocr.Reader(['en'])
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        self.is_running = True
        self.monitoring_active = False
        self.last_registration_time = time.time()

        # Setup CustomTkinter UI
        self.setup_ui()
        
    def log_access(self, identity, matric, role, entry_type):
        cursor.execute("SELECT timestamp FROM access_log WHERE identity = %s ORDER BY timestamp DESC LIMIT 1", (identity,))
        last_entry = cursor.fetchone()

        # ✅ Prevent duplicate entry (Cooldown: 10 seconds)
        if last_entry:
            last_timestamp = last_entry[0]
            if datetime.now() - last_timestamp < timedelta(seconds=10):
                print(f"Skipping duplicate entry. Last entry timestamp: {last_timestamp}")
                return  # Skip registration if less than 10 seconds have passed

        # ✅ Insert entry into access_log
        cursor.execute("INSERT INTO access_log (identity, matric, role, entry_type, timestamp) VALUES (%s, %s, %s, %s, NOW())", 
                       (identity, matric, role, entry_type))
        db.commit()
        print(f"Inserted new access log for {identity} with type {entry_type}.")

    def setup_ui(self):
        """ Setup UI with CustomTkinter """
        # Sidebar
        self.sidebar = ctk.CTkFrame(self.root, width=250, corner_radius=10, bg_color="#1E1E1E")
        self.sidebar.pack(side="left", fill="y")

        self.logo_label = ctk.CTkLabel(self.sidebar, text="🚔 UNISEL SECURITY", font=("Arial", 16, "bold"), text_color="white", bg_color="#1E1E1E")
        self.logo_label.pack(pady=20)

        button_texts = ["📹 Monitoring"]
        for text in button_texts:
            btn = ctk.CTkButton(self.sidebar, text=text, font=("Arial", 16), text_color="white", bg_color="#202020", height=60, width=230, command=lambda t=text: self.switch_content(t))
            btn.pack(pady=20, padx=30, fill="x")

        # Add a line between sidebar and content area
        self.separator_line = ctk.CTkFrame(self.root, width=2, corner_radius=0, bg_color="#333333")
        self.separator_line.pack(side="left", fill="y")

        # Content area
        self.content_frame = ctk.CTkFrame(self.root, corner_radius=10, bg_color="#1E1E1E")
        self.content_frame.pack(side="right", fill="both", expand=True)

    def switch_content(self, text):
        """ Switch between the sections (Monitoring) """
        for widget in self.content_frame.winfo_children():
            widget.destroy()

        if text == "📹 Monitoring":
            self.init_monitoring()

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
            print("You can only register an unrecognized face once every 10 seconds.")
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
                print("⚠ No camera feed available.")
        else:
            print("⚠ camera_label widget no longer exists.")

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

# Run the app
if __name__ == "__main__":
    root = ctk.CTk()
    app = SecurityStaff(root)
    root.mainloop()
