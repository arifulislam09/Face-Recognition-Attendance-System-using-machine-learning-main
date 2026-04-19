import os
import sqlite3
import csv
import io
import numpy as np
import cv2
import face_recognition
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey123"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "images")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)


def get_db():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            image_path TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    cur.execute("SELECT id FROM Users WHERE username = ?", ("admin",))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO Users (username, password) VALUES (?, ?)",
            ("admin", "admin123"),
        )
    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_known_faces():
    conn = get_db()
    rows = conn.execute("SELECT * FROM Students").fetchall()
    conn.close()
    known_encodings = []
    known_labels = []

    for row in rows:
        file_path = os.path.join(BASE_DIR, row["image_path"])
        if not os.path.exists(file_path):
            continue

        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_labels.append((row["student_id"], row["name"]))

    return known_encodings, known_labels


def mark_attendance(student_id, name):
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    existing = conn.execute(
        "SELECT * FROM Attendance WHERE student_id = ? AND date = ?",
        (student_id, today),
    ).fetchone()
    if existing:
        conn.close()
        return False

    conn.execute(
        "INSERT INTO Attendance (student_id, name, date, time, status) VALUES (?, ?, ?, ?, ?)",
        (student_id, name, today, datetime.now().strftime("%H:%M:%S"), "Present"),
    )
    conn.commit()
    conn.close()
    return True


@app.route("/")
def home():
    if session.get("user"):
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Please enter both username and password.", "warning")
            return redirect(url_for("login"))

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM Users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()

        if user:
            session["user"] = username
            flash("Login successful. Welcome back!", "success")
            return redirect(url_for("dashboard"))

        flash("Login failed. Check your username and password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login"))

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    student_count = conn.execute("SELECT COUNT(*) FROM Students").fetchone()[0]
    total_records = conn.execute("SELECT COUNT(*) FROM Attendance").fetchone()[0]
    today_records = conn.execute(
        "SELECT COUNT(*) FROM Attendance WHERE date = ?", (today,)
    ).fetchone()[0]
    recent_students = conn.execute(
        "SELECT * FROM Students ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        student_count=student_count,
        total_records=total_records,
        today_records=today_records,
        recent_students=recent_students,
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if not session.get("user"):
        return redirect(url_for("login"))

    conn = get_db()
    students = conn.execute("SELECT * FROM Students ORDER BY id DESC").fetchall()
    conn.close()

    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip()
        image = request.files.get("image")

        if not student_id or not name or not department or not image:
            flash("All fields are required, including the student photo.", "warning")
            return redirect(url_for("register"))

        if not allowed_file(image.filename):
            flash("Only JPG, JPEG, and PNG images are allowed.", "danger")
            return redirect(url_for("register"))

        filename = secure_filename(f"{student_id}_{name}.{image.filename.rsplit('.', 1)[1].lower()}")
        image_path = os.path.join("static", "images", filename)
        full_image_path = os.path.join(BASE_DIR, image_path)
        dataset_path = os.path.join(DATASET_FOLDER, filename)

        try:
            image.save(full_image_path)
            image.save(dataset_path)

            conn = get_db()
            conn.execute(
                "INSERT INTO Students (student_id, name, department, image_path) VALUES (?, ?, ?, ?)",
                (student_id, name, department, image_path),
            )
            conn.commit()
            conn.close()
            flash("Student registered successfully.", "success")
            return redirect(url_for("register"))
        except sqlite3.IntegrityError:
            flash("This student ID already exists. Use a different ID.", "danger")
            return redirect(url_for("register"))
        except Exception as exc:
            flash(f"Error saving student: {exc}", "danger")
            return redirect(url_for("register"))

    return render_template("register.html", students=students)


@app.route("/delete_student/<int:student_id>")
def delete_student(student_id):
    if not session.get("user"):
        return redirect(url_for("login"))

    conn = get_db()
    row = conn.execute("SELECT * FROM Students WHERE id = ?", (student_id,)).fetchone()
    if row:
        image_file = os.path.join(BASE_DIR, row["image_path"])
        if os.path.exists(image_file):
            os.remove(image_file)
        conn.execute("DELETE FROM Students WHERE id = ?", (student_id,))
        conn.commit()
        flash("Student removed successfully.", "success")
    else:
        flash("Student not found.", "warning")
    conn.close()
    return redirect(url_for("register"))


@app.route("/scanner")
def scanner():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("scanner.html")


@app.route("/start_scan", methods=["POST"])
def start_scan():
    if not session.get("user"):
        return redirect(url_for("login"))

    known_encodings, known_labels = load_known_faces()
    if not known_encodings:
        flash("No registered student faces found. Add students before scanning.", "warning")
        return redirect(url_for("scanner"))

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        flash("Unable to access the webcam. Check your camera and try again.", "danger")
        return redirect(url_for("scanner"))

    scanned_students = set()
    start_time = datetime.now()
    scan_duration = 18

    while (datetime.now() - start_time).seconds < scan_duration:
        grabbed, frame = video.read()
        if not grabbed:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) == 0:
                continue

            best_index = int(np.argmin(distances))
            match = face_recognition.compare_faces([known_encodings[best_index]], face_encoding, tolerance=0.5)

            if match and match[0]:
                student_id, name = known_labels[best_index]
                if student_id not in scanned_students:
                    scanned_students.add(student_id)
                    if mark_attendance(student_id, name):
                        flash(f"Marked attendance for {name}.", "success")
                    else:
                        flash(f"{name} already has attendance today.", "info")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()

    if not scanned_students:
        flash("No faces matched during the scan. Try again with better lighting.", "warning")

    return redirect(url_for("scanner"))


@app.route("/records")
def records():
    if not session.get("user"):
        return redirect(url_for("login"))

    search_date = request.args.get("date", "").strip()
    conn = get_db()

    if search_date:
        records = conn.execute(
            "SELECT * FROM Attendance WHERE date = ? ORDER BY id DESC",
            (search_date,),
        ).fetchall()
    else:
        records = conn.execute("SELECT * FROM Attendance ORDER BY id DESC").fetchall()

    conn.close()
    return render_template("records.html", records=records, search_date=search_date)


@app.route("/download_csv")
def download_csv():
    if not session.get("user"):
        return redirect(url_for("login"))

    search_date = request.args.get("date", "").strip()
    conn = get_db()
    if search_date:
        records = conn.execute(
            "SELECT * FROM Attendance WHERE date = ? ORDER BY id DESC",
            (search_date,),
        ).fetchall()
    else:
        records = conn.execute("SELECT * FROM Attendance ORDER BY id DESC").fetchall()
    conn.close()

    if not records:
        flash("No attendance data available to download.", "warning")
        return redirect(url_for("records"))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Date", "Time", "Status"])
    for row in records:
        writer.writerow([row["student_id"], row["name"], row["date"], row["time"], row["status"]])

    csv_data = BytesIO()
    csv_data.write(output.getvalue().encode("utf-8"))
    csv_data.seek(0)

    return send_file(
        csv_data,
        mimetype="text/csv",
        download_name="attendance_report.csv",
        as_attachment=True,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
