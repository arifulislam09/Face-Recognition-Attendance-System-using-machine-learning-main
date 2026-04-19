import os
import sqlite3
import csv
import io
import requests
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey123"

# Gmail OAuth Config (Update with your Google OAuth credentials)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "")

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "images")
DATASET_FOLDER = os.path.join(BASE_DIR, "dataset")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

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
            password TEXT NOT NULL,
            role TEXT DEFAULT 'student',
            full_name TEXT,
            department TEXT,
            approval_status TEXT DEFAULT 'pending',
            created_at TEXT,
            approved_at TEXT,
            approved_by TEXT,
            auth_provider TEXT DEFAULT 'password'
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
            "INSERT INTO Users (username, password, role, full_name, approval_status, created_at, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("admin", "admin123", "admin", "System Admin", "approved", datetime.now().isoformat(), "password"),
        )

    # Lightweight migration so existing databases also support approval workflow.
    user_columns = {row[1] for row in cur.execute("PRAGMA table_info(Users)").fetchall()}
    required_columns = {
        "full_name": "TEXT",
        "department": "TEXT",
        "approval_status": "TEXT DEFAULT 'pending'",
        "created_at": "TEXT",
        "approved_at": "TEXT",
        "approved_by": "TEXT",
        "auth_provider": "TEXT DEFAULT 'password'",
    }
    for column_name, column_type in required_columns.items():
        if column_name not in user_columns:
            cur.execute(f"ALTER TABLE Users ADD COLUMN {column_name} {column_type}")

    cur.execute(
        "UPDATE Users SET role = 'student' WHERE role = 'user'"
    )
    cur.execute(
        "UPDATE Users SET approval_status = 'approved', auth_provider = COALESCE(auth_provider, 'password') WHERE username = 'admin'"
    )
    cur.execute(
        "UPDATE Users SET approval_status = COALESCE(approval_status, 'pending'), created_at = COALESCE(created_at, ?) WHERE username != 'admin'",
        (datetime.now().isoformat(),),
    )

    conn.commit()
    conn.close()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_google_redirect_uri():
    if GOOGLE_REDIRECT_URI:
        return GOOGLE_REDIRECT_URI
    return url_for("google_callback", _external=True)


def is_student_pending(user_row):
    return user_row["role"] == "student" and user_row["approval_status"] != "approved"


def has_lbph_support():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def load_face_recognizer():
    if not has_lbph_support():
        return None, {}

    conn = get_db()
    rows = conn.execute("SELECT * FROM Students").fetchall()
    conn.close()

    face_samples = []
    labels = []
    label_map = {}

    for row in rows:
        file_path = os.path.join(BASE_DIR, row["image_path"])
        if not os.path.exists(file_path):
            continue

        image = cv2.imread(file_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        face_region = gray[y : y + h, x : x + w]

        label = int(row["id"])
        face_samples.append(face_region)
        labels.append(label)
        label_map[label] = (row["student_id"], row["name"])

    if not face_samples:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(labels))
    return recognizer, label_map


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
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("user_dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user"):
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("user_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        login_portal = request.form.get("login_portal", "").strip().lower()

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
            if login_portal in {"admin", "student"} and user["role"] != login_portal:
                flash(f"This account is not registered for {login_portal} portal login.", "danger")
                return redirect(url_for("login"))

            if is_student_pending(user):
                flash("Your student account is pending admin approval. Please wait for approval.", "warning")
                return redirect(url_for("login"))

            session["user"] = username
            session["role"] = user["role"]
            flash(f"Login successful. Welcome back, {user['role']}!", "success")
            if user["role"] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("user_dashboard"))

        flash("Login failed. Check your username and password.", "danger")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/auth/google")
@app.route("/login/google")
def google_login():
    if not GOOGLE_CLIENT_ID:
        flash("Google login is not configured. Please set GOOGLE_CLIENT_ID.", "warning")
        return redirect(url_for("login"))

    google_discovery_url = GOOGLE_DISCOVERY_URL
    discovery = requests.get(google_discovery_url).json()
    auth_endpoint = discovery["authorization_endpoint"]

    redirect_uri = get_google_redirect_uri()
    request_uri = requests.Request(
        "GET",
        auth_endpoint,
        params={
            "client_id": GOOGLE_CLIENT_ID,
            "response_type": "code",
            "scope": "openid email profile",
            "redirect_uri": redirect_uri,
            "access_type": "online",
            "prompt": "select_account",
        },
    ).prepare().url
    return redirect(request_uri)


@app.route("/auth/google/callback")
def google_callback():
    code = request.args.get("code")
    if not code or not GOOGLE_CLIENT_ID:
        flash("Google authentication failed.", "danger")
        return redirect(url_for("login"))

    try:
        google_discovery_url = GOOGLE_DISCOVERY_URL
        discovery = requests.get(google_discovery_url).json()
        token_endpoint = discovery["token_endpoint"]

        redirect_uri = get_google_redirect_uri()
        token_request_data = {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }

        token_response = requests.post(token_endpoint, data=token_request_data, timeout=15)
        token_response.raise_for_status()
        token_data = token_response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("No access token returned from Google.")

        user_info_url = discovery["userinfo_endpoint"]
        user_info = requests.get(
            user_info_url,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        user_info.raise_for_status()
        user_data = user_info.json()

        email = user_data.get("email", "")
        name = user_data.get("name", email)

        if not email:
            flash("Unable to get email from Google. Please try again.", "danger")
            return redirect(url_for("login"))

        conn = get_db()
        user = conn.execute("SELECT * FROM Users WHERE username = ?", (email,)).fetchone()

        if not user:
            conn.execute(
                "INSERT INTO Users (username, password, role, full_name, approval_status, created_at, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (email, "google_oauth", "student", name, "pending", datetime.now().isoformat(), "google"),
            )
            conn.commit()
            conn.close()
            session["pending_registration_user"] = email
            flash("Google login successful. Complete student registration for admin approval.", "info")
            return redirect(url_for("student_registration_request"))

        if is_student_pending(user):
            conn.close()
            session["pending_registration_user"] = email
            flash("Your account is pending. Complete your registration details and wait for admin approval.", "warning")
            return redirect(url_for("student_registration_request"))

        conn.close()

        session["user"] = email
        session["role"] = user["role"]
        flash(f"Login successful. Welcome, {name}!", "success")
        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))

    except Exception as e:
        flash(f"Google authentication error: {str(e)}", "danger")
        return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user"):
        if session.get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("user_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not full_name or not department or not password or not confirm_password:
            flash("Please complete all fields.", "warning")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return redirect(url_for("signup"))

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO Users (username, password, role, full_name, department, approval_status, created_at, auth_provider) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (username, password, "student", full_name, department, "pending", datetime.now().isoformat(), "password"),
            )
            conn.commit()
            conn.close()
            flash("Registration submitted. Wait for admin approval before login.", "info")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Username already exists. Choose a different username.", "danger")
            return redirect(url_for("signup"))

    return render_template("signup.html")


@app.route("/student-registration", methods=["GET", "POST"])
def student_registration_request():
    pending_email = session.get("pending_registration_user")
    if not pending_email:
        flash("Start with Google login to submit a student registration request.", "warning")
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM Users WHERE username = ?", (pending_email,)).fetchone()
    if not user:
        conn.close()
        session.pop("pending_registration_user", None)
        flash("Registration session expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    if user["approval_status"] == "approved":
        conn.close()
        session.pop("pending_registration_user", None)
        session["user"] = user["username"]
        session["role"] = user["role"]
        flash("Your account is already approved. Welcome back!", "success")
        return redirect(url_for("user_dashboard"))

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        department = request.form.get("department", "").strip()

        if not full_name or not department:
            flash("Please enter your full name and department.", "warning")
            conn.close()
            return redirect(url_for("student_registration_request"))

        conn.execute(
            "UPDATE Users SET full_name = ?, department = ?, approval_status = 'pending', created_at = COALESCE(created_at, ?), auth_provider = COALESCE(auth_provider, 'google') WHERE id = ?",
            (full_name, department, datetime.now().isoformat(), user["id"]),
        )
        conn.commit()
        conn.close()
        session.pop("pending_registration_user", None)
        flash("Registration request submitted. Admin approval is required.", "info")
        return redirect(url_for("login"))

    conn.close()
    return render_template("student_registration.html", user=user)


@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("role", None)
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("user") or session.get("role") != "admin":
        return redirect(url_for("login"))

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    student_count = conn.execute("SELECT COUNT(*) FROM Students").fetchone()[0]
    total_student_accounts = conn.execute(
        "SELECT COUNT(*) FROM Users WHERE role = 'student'"
    ).fetchone()[0]
    approved_student_accounts = conn.execute(
        "SELECT COUNT(*) FROM Users WHERE role = 'student' AND approval_status = 'approved'"
    ).fetchone()[0]
    pending_student_requests = conn.execute(
        "SELECT COUNT(*) FROM Users WHERE role = 'student' AND approval_status = 'pending'"
    ).fetchone()[0]
    pending_users = conn.execute(
        "SELECT id, username, full_name, department, created_at FROM Users WHERE role = 'student' AND approval_status = 'pending' ORDER BY id DESC"
    ).fetchall()
    approved_users = conn.execute(
        "SELECT username, full_name, department, approved_at, approved_by FROM Users WHERE role = 'student' AND approval_status = 'approved' ORDER BY id DESC LIMIT 10"
    ).fetchall()
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
        total_student_accounts=total_student_accounts,
        approved_student_accounts=approved_student_accounts,
        pending_student_requests=pending_student_requests,
        pending_users=pending_users,
        approved_users=approved_users,
        total_records=total_records,
        today_records=today_records,
        recent_students=recent_students,
        is_admin=True,
    )


@app.route("/user_dashboard")
def user_dashboard():
    if not session.get("user") or session.get("role") != "student":
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM Users WHERE username = ?", (session["user"],)).fetchone()
    if not user or is_student_pending(user):
        conn.close()
        session.pop("user", None)
        session.pop("role", None)
        flash("Your account is pending admin approval.", "warning")
        return redirect(url_for("login"))

    today = datetime.now().strftime("%Y-%m-%d")
    total_records = conn.execute("SELECT COUNT(*) FROM Attendance").fetchone()[0]
    today_records = conn.execute(
        "SELECT COUNT(*) FROM Attendance WHERE date = ?", (today,)
    ).fetchone()[0]
    conn.close()

    return render_template(
        "dashboard.html",
        student_count=0,
        total_student_accounts=0,
        approved_student_accounts=0,
        pending_student_requests=0,
        pending_users=[],
        approved_users=[],
        total_records=total_records,
        today_records=today_records,
        recent_students=[],
        is_admin=False,
    )


@app.route("/approve_user/<int:user_id>")
def approve_user(user_id):
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can approve students.", "danger")
        return redirect(url_for("login"))

    conn = get_db()
    user = conn.execute("SELECT * FROM Users WHERE id = ?", (user_id,)).fetchone()
    if not user or user["role"] != "student":
        conn.close()
        flash("Student request not found.", "warning")
        return redirect(url_for("admin_dashboard"))

    conn.execute(
        "UPDATE Users SET approval_status = 'approved', approved_at = ?, approved_by = ? WHERE id = ?",
        (datetime.now().isoformat(), session.get("user", "admin"), user_id),
    )
    conn.commit()
    conn.close()
    flash("Student registration approved successfully.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/dashboard")
def dashboard():
    if session.get("role") == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("user_dashboard"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can register students.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

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
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can delete students.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

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
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can access the scanner.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

    cloud_hosted = os.environ.get("RENDER", "").lower() == "true"
    return render_template("scanner.html", cloud_hosted=cloud_hosted)


@app.route("/start_scan", methods=["POST"])
def start_scan():
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can start scanning.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

    if os.environ.get("RENDER", "").lower() == "true":
        flash("Webcam-based scanning is not available on Render. Run the scanner locally.", "warning")
        return redirect(url_for("scanner"))

    if not has_lbph_support():
        flash(
            "Face recognizer is unavailable. Install opencv-contrib-python (or opencv-contrib-python-headless in server environments).",
            "danger",
        )
        return redirect(url_for("scanner"))

    recognizer, label_map = load_face_recognizer()
    if recognizer is None:
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

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_region = gray_frame[y : y + h, x : x + w]
            try:
                label, confidence = recognizer.predict(face_region)
            except cv2.error:
                continue

            if confidence < 70:
                student_id, name = label_map.get(label, (None, None))
                if student_id and student_id not in scanned_students:
                    scanned_students.add(student_id)
                    if mark_attendance(student_id, name):
                        flash(f"Marked attendance for {name}.", "success")
                    else:
                        flash(f"{name} already has attendance today.", "info")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_info = label_map.get(label)
            text = f"{label_info[1]} ({int(confidence)})" if label_info else f"Unknown ({int(confidence)})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Attendance Scanner", frame)
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

init_db()
