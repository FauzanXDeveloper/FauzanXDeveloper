import mysql.connector
import uuid
from mysql.connector import Error

def create_database():
    try:
        # Connect to MySQL Server
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Change if you have a different user
            password=""    # Add your MySQL password if needed
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS unisel_security;")
            print("Database 'unisel_security' created successfully!")
        
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def create_tables():
    try:
        # Connect to unisel_security Database
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="unisel_security"
        )
        
        if connection.is_connected():
            cursor = connection.cursor()
            
            # ✅ access_log Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS access_log (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                entry_type ENUM('Face','Vehicle') NOT NULL,
                identity VARCHAR(100) NOT NULL,
                matric VARCHAR(20) NOT NULL,
                role ENUM('Student','Staff') NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );''')
            
            # ✅ staff Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS staff (
                staff_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                matric VARCHAR(20) UNIQUE NOT NULL,
                role ENUM('Staff') DEFAULT 'Staff',
                staff_image LONGBLOB NOT NULL
            );''')
            
            # ✅ students Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                student_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100),
                matric VARCHAR(20) UNIQUE NOT NULL,
                role ENUM('Student') DEFAULT 'Student',
                student_image LONGBLOB NOT NULL
            );''')
            
            # ✅ unrecognized_faces Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS unrecognized_faces (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image LONGBLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                identity VARCHAR(255),
                camera VARCHAR(50) DEFAULT 'Main Gate'
            );''')
            
            # ✅ unrecognized_plates Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS unrecognized_plates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image LONGBLOB NOT NULL,
                plate_number VARCHAR(15),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera VARCHAR(50) DEFAULT 'Main Gate'
            );''')
            
            # ✅ users Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(100) NOT NULL
            );''')
            
            # ✅ vehicles Table
            cursor.execute('''CREATE TABLE IF NOT EXISTS vehicles (
                vehicle_id INT AUTO_INCREMENT PRIMARY KEY,
                owner_name VARCHAR(100) NOT NULL,
                matric VARCHAR(20) NOT NULL,
                role ENUM('Student','Staff'),
                plate_number VARCHAR(15) UNIQUE NOT NULL,
                plate_image LONGBLOB NOT NULL,
                owner_image LONGBLOB NOT NULL
            );''')
            
            connection.commit()
            print("All tables created successfully!")
        
    except Error as e:
        print("Error while creating tables", e)
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    create_database()
    create_tables()
