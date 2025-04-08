import mysql.connector
from mysql.connector import Error

def connect_db():
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='unisel_security',
            autocommit=True
        )
        return connection
    except Error as e:
        print("Error connecting to MySQL:", e)
        return None
