from flask import Flask, render_template
import mysql.connector

app = Flask(__name__)

# Connect to MySQL
def get_logs():
    db = mysql.connector.connect(host="localhost", user="root", passwd="", database="unisel_security")
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM access_log ORDER BY timestamp DESC LIMIT 20")  # Last 20 logs
    logs = cursor.fetchall()
    db.close()
    return logs

@app.route("/")
def home():
    logs = get_logs()
    return render_template("dashboard.html", logs=logs)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
