from sjvisualizer import Canvas
from sjvisualizer import DataHandler
from sjvisualizer import PieRace
import tkinter as tk
import mysql.connector
import time
import json

class UNISELSecuritySystem:
    def __init__(self):
        self.db_connection = self.connect_db()

    def connect_db(self):
        """Establishes connection to MySQL database"""
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="unisel_security"
            )
            return connection
        except mysql.connector.Error as e:
            print(f"Error connecting to database: {e}")
            return None

    def get_role_data(self):
        """Fetch role data from the database"""
        query = "SELECT role, COUNT(*) FROM access_log GROUP BY role"
        cursor = self.db_connection.cursor()
        cursor.execute(query)
        data = cursor.fetchall()
        
        roles, counts = [], []
        for row in data:
            roles.append(row[0])
            counts.append(row[1])
        
        return roles, counts

    def display_role_chart(self, parent_frame):
        """Generates role-based access report chart with animated pie chart"""
        roles, counts = self.get_role_data()

        # Create DataHandler with roles and their corresponding counts
        data = [{"role": roles[i], "count": counts[i]} for i in range(len(roles))]
        
        # Create Canvas for Pie Race visualization
        canvas = Canvas(width=500, height=500)

        # PieRace: Create the pie chart animation with custom configuration
        pie_race = PieRace(
            canvas,
            data,
            title="Role-Based Access Distribution",
            field="role",
            value="count",
            colors=["#3498db", "#2ECC71"],  # Colors for "Student" and "Staff"
            speed=50,  # Speed of the animation
            start_angle=-90,  # Start the pie chart from top
            end_angle=270,  # Rotate until 270 degrees
            interval=100  # Interval for smooth animation
        )
        
        # Start the animation
        pie_race.animate()
        
        # Embed the pie chart into the parent_frame (Tkinter frame)
        canvas.pack(fill="both", expand=True, pady=10)

# Assuming this is part of your Tkinter app and you want to display the pie chart in a frame
root = tk.Tk()
app = UNISELSecuritySystem()
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)
app.display_role_chart(frame)
root.mainloop()
