import os
import sqlite3
import csv
import io
import requests
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

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
MODELS_DIR = os.path.join(BASE_DIR, "models")
LBF_LANDMARK_MODEL = os.path.join(MODELS_DIR, "lbfmodel.yaml")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDERS = ["Male", "Female"]

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
    configured_uri = (GOOGLE_REDIRECT_URI or "").strip()

    # On local development, keep callback host aligned with the current request host
    # so localhost and 127.0.0.1 do not conflict.
    current_host = request.host.split(":", 1)[0].lower()
    if current_host in {"localhost", "127.0.0.1"}:
        return url_for("google_callback", _external=True)

    if configured_uri:
        return configured_uri
    return url_for("google_callback", _external=True)


def is_student_pending(user_row):
    return user_row["role"] == "student" and user_row["approval_status"] != "approved"


def has_lbph_support():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def has_facemark_support():
    return hasattr(cv2, "face") and hasattr(cv2.face, "createFacemarkLBF")


def load_facemark_model():
    if not has_facemark_support() or not os.path.exists(LBF_LANDMARK_MODEL):
        return None

    try:
        facemark = cv2.face.createFacemarkLBF()
        facemark.loadModel(LBF_LANDMARK_MODEL)
        logger.info("Loaded LBF landmark model successfully.")
        return facemark
    except cv2.error as exc:
        logger.warning(f"Unable to load facial landmark model: {exc}")
        return None


def load_demographics_models():
    if not all(os.path.exists(path) for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]):
        return None, None

    try:
        age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        logger.info("Loaded age/gender models successfully.")
        return age_net, gender_net
    except cv2.error as exc:
        logger.warning(f"Unable to load age/gender models: {exc}")
        return None, None


class SimpleFaceTracker:
    def __init__(self, max_disappeared=15, distance_threshold=80):
        self.next_track_id = 1
        self.tracks = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    @staticmethod
    def _centroid(rect):
        x, y, w, h = rect
        return np.array([x + (w / 2.0), y + (h / 2.0)], dtype=np.float32)

    def _register(self, centroid):
        track_id = self.next_track_id
        self.tracks[track_id] = centroid
        self.disappeared[track_id] = 0
        self.next_track_id += 1
        return track_id

    def _deregister(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]

    def update(self, rects):
        if not rects:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self._deregister(track_id)
            return []

        input_centroids = np.array([self._centroid(rect) for rect in rects], dtype=np.float32)
        assigned_track_ids = [None] * len(rects)

        if len(self.tracks) == 0:
            for idx in range(len(rects)):
                assigned_track_ids[idx] = self._register(input_centroids[idx])
            return assigned_track_ids

        track_ids = list(self.tracks.keys())
        track_centroids = np.array(list(self.tracks.values()), dtype=np.float32)
        distances = np.linalg.norm(track_centroids[:, None, :] - input_centroids[None, :, :], axis=2)

        used_track_rows = set()
        used_detection_cols = set()
        sorted_indices = np.unravel_index(np.argsort(distances, axis=None), distances.shape)

        for row, col in zip(sorted_indices[0], sorted_indices[1]):
            if row in used_track_rows or col in used_detection_cols:
                continue
            if distances[row, col] > self.distance_threshold:
                continue

            track_id = track_ids[row]
            self.tracks[track_id] = input_centroids[col]
            self.disappeared[track_id] = 0
            assigned_track_ids[col] = track_id
            used_track_rows.add(row)
            used_detection_cols.add(col)

        unused_track_rows = set(range(len(track_ids))) - used_track_rows
        for row in unused_track_rows:
            track_id = track_ids[row]
            self.disappeared[track_id] += 1
            if self.disappeared[track_id] > self.max_disappeared:
                self._deregister(track_id)

        unused_detection_cols = set(range(len(rects))) - used_detection_cols
        for col in unused_detection_cols:
            assigned_track_ids[col] = self._register(input_centroids[col])

        return assigned_track_ids


def detect_landmarks(frame_gray, faces, facemark):
    if facemark is None or not faces:
        return {}

    try:
        face_array = np.array([[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces], dtype=np.int32)
        ok, landmarks = facemark.fit(frame_gray, face_array)
        if not ok:
            return {}
        return {idx: lm[0] for idx, lm in enumerate(landmarks)}
    except cv2.error as exc:
        logger.debug(f"Landmark detection skipped: {exc}")
        return {}


def _safe_crop(gray_frame, rect):
    x, y, w, h = rect
    x = max(0, x)
    y = max(0, y)
    w = max(1, w)
    h = max(1, h)
    x2 = min(gray_frame.shape[1], x + w)
    y2 = min(gray_frame.shape[0], y + h)
    return gray_frame[y:y2, x:x2]


def align_and_normalize_face(gray_frame, rect, landmarks=None, target_size=(200, 200)):
    x, y, w, h = rect
    aligned_frame = gray_frame

    if landmarks is not None and len(landmarks) >= 48:
        left_eye_center = landmarks[36:42].mean(axis=0)
        right_eye_center = landmarks[42:48].mean(axis=0)
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = float(np.degrees(np.arctan2(dy, dx)))
        eye_midpoint = tuple(((left_eye_center + right_eye_center) / 2.0).astype(np.float32))
        rotation_matrix = cv2.getRotationMatrix2D(eye_midpoint, angle, 1.0)
        aligned_frame = cv2.warpAffine(gray_frame, rotation_matrix, (gray_frame.shape[1], gray_frame.shape[0]))

    face = _safe_crop(aligned_frame, (x, y, w, h))
    if face.size == 0:
        face = _safe_crop(gray_frame, (x, y, w, h))

    face = cv2.equalizeHist(face)
    return cv2.resize(face, target_size)


def eye_aspect_ratio(eye_points):
    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
    if p1_p4 == 0:
        return 0.0
    return float((p2_p6 + p3_p5) / (2.0 * p1_p4))


def estimate_emotion(face_gray):
    mean_intensity = float(np.mean(face_gray))
    texture = float(np.std(face_gray))
    edges = float(np.mean(cv2.Canny(face_gray, 80, 160)))

    if mean_intensity > 145 and texture > 38:
        return "Happy", 0.68
    if mean_intensity < 95 and texture < 35:
        return "Sad", 0.62
    if edges > 42:
        return "Surprised", 0.60
    if texture < 22:
        return "Calm", 0.58
    return "Neutral", 0.65


def estimate_demographics(face_bgr, age_net, gender_net):
    if age_net is None or gender_net is None:
        return "Unknown", "Unknown"

    try:
        face = cv2.resize(face_bgr, (227, 227))
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        gender_net.setInput(blob)
        gender_preds = gender_net.forward()[0]
        gender = GENDERS[int(np.argmax(gender_preds))]

        age_net.setInput(blob)
        age_preds = age_net.forward()[0]
        age = AGE_BUCKETS[int(np.argmax(age_preds))]
        return age, gender
    except cv2.error as exc:
        logger.debug(f"Demographic analysis skipped: {exc}")
        return "Unknown", "Unknown"


def estimate_liveness(track_state, face_gray, landmarks=None):
    downscaled = cv2.resize(face_gray, (64, 64))
    prev = track_state.get("prev_face")
    motion = 0.0

    if prev is not None and prev.shape == downscaled.shape:
        motion = float(np.mean(cv2.absdiff(prev, downscaled)))

    track_state["prev_face"] = downscaled

    blink_detected = False
    if landmarks is not None and len(landmarks) >= 48:
        left_ear = eye_aspect_ratio(landmarks[36:42])
        right_ear = eye_aspect_ratio(landmarks[42:48])
        ear = (left_ear + right_ear) / 2.0
        previous_ear = track_state.get("prev_ear", ear)
        track_state["prev_ear"] = ear
        if previous_ear > 0.24 and ear < 0.20:
            blink_detected = True

    liveness_score = min(1.0, motion / 18.0)
    if blink_detected:
        liveness_score = min(1.0, liveness_score + 0.30)

    if liveness_score >= 0.45:
        return "Live", liveness_score
    return "Spoof risk", liveness_score


def parse_camera_indexes(raw_value):
    indexes = []
    if not raw_value:
        return [0, 1, 2]

    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            index = int(part)
        except ValueError:
            continue
        if index < 0:
            continue
        if index not in indexes:
            indexes.append(index)

    return indexes or [0, 1, 2]


def build_camera_candidates(camera_source, camera_url, camera_indexes_raw):
    if camera_source == "phone_url":
        if not camera_url:
            return []
        return [("Phone camera URL", camera_url)]

    if camera_source == "usb":
        return [
            ("External/virtual camera (index 1)", 1),
            ("External/virtual camera (index 2)", 2),
            ("Built-in camera (index 0)", 0),
        ]

    if camera_source == "multiple":
        indexes = parse_camera_indexes(camera_indexes_raw)
        return [(f"Camera index {index}", index) for index in indexes]

    return [("Built-in camera (index 0)", 0)]


def open_camera(candidates):
    for label, source in candidates:
        logger.info(f"Trying camera: {label} (source={source})")
        video = cv2.VideoCapture(source)
        if video.isOpened():
            logger.info(f"Successfully opened: {label}")
            return video, label
        video.release()
        logger.warning(f"Failed to open: {label}")
    return None, None


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
        face_region_resized = cv2.resize(face_region, (200, 200))

        label = int(row["id"])
        face_samples.append(face_region_resized)
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

    trend_days = [(datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(6, -1, -1)]
    trend_map = {day: 0 for day in trend_days}
    trend_rows = conn.execute(
        "SELECT date, COUNT(*) AS total FROM Attendance WHERE date >= ? AND date <= ? GROUP BY date",
        (trend_days[0], trend_days[-1]),
    ).fetchall()
    for row in trend_rows:
        trend_map[row["date"]] = row["total"]

    chart_labels = [datetime.strptime(day, "%Y-%m-%d").strftime("%d %b") for day in trend_days]
    chart_values = [trend_map[day] for day in trend_days]

    dept_rows = conn.execute(
        """
        SELECT COALESCE(Students.department, 'Unassigned') AS department, COUNT(*) AS total
        FROM Attendance
        LEFT JOIN Students ON Students.student_id = Attendance.student_id
        GROUP BY COALESCE(Students.department, 'Unassigned')
        ORDER BY total DESC
        LIMIT 6
        """
    ).fetchall()
    dept_labels = [row["department"] for row in dept_rows]
    dept_values = [row["total"] for row in dept_rows]

    recent_students = conn.execute(
        "SELECT * FROM Students ORDER BY id DESC LIMIT 5"
    ).fetchall()
    conn.close()

    capability_status = {
        "real_time_boxes": True,
        "facial_landmarks": has_facemark_support() and os.path.exists(LBF_LANDMARK_MODEL),
        "face_tracking": True,
        "multiple_faces": True,
        "alignment_normalization": True,
        "emotion_recognition": True,
        "demographic_analysis": all(
            os.path.exists(path) for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
        ),
        "liveness_detection": True,
    }
    capability_items = [
        ("real_time_boxes", "Real-time bounding boxes"),
        ("facial_landmarks", "Facial landmark localization"),
        ("face_tracking", "Face tracking IDs"),
        ("multiple_faces", "Multiple face detection"),
        ("alignment_normalization", "Alignment and normalization"),
        ("emotion_recognition", "Emotion recognition"),
        ("demographic_analysis", "Demographic analysis"),
        ("liveness_detection", "Liveness detection"),
    ]

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
        chart_labels=chart_labels,
        chart_values=chart_values,
        dept_labels=dept_labels,
        dept_values=dept_values,
        recent_students=recent_students,
        capability_status=capability_status,
        capability_items=capability_items,
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
    student_names = [value for value in [user["full_name"], user["username"]] if value]
    student_names = list(dict.fromkeys(student_names))

    total_records = 0
    today_records = 0
    recent_attendance = []

    if student_names:
        placeholders = ",".join("?" * len(student_names))
        total_records = conn.execute(
            f"SELECT COUNT(*) FROM Attendance WHERE name IN ({placeholders})",
            student_names,
        ).fetchone()[0]
        today_records = conn.execute(
            f"SELECT COUNT(*) FROM Attendance WHERE date = ? AND name IN ({placeholders})",
            [today, *student_names],
        ).fetchone()[0]
        recent_attendance = conn.execute(
            f"SELECT * FROM Attendance WHERE name IN ({placeholders}) ORDER BY id DESC LIMIT 5",
            student_names,
        ).fetchall()

    conn.close()

    capability_status = {
        "real_time_boxes": True,
        "facial_landmarks": has_facemark_support() and os.path.exists(LBF_LANDMARK_MODEL),
        "face_tracking": True,
        "multiple_faces": True,
        "alignment_normalization": True,
        "emotion_recognition": True,
        "demographic_analysis": all(
            os.path.exists(path) for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
        ),
        "liveness_detection": True,
    }
    capability_items = [
        ("real_time_boxes", "Real-time bounding boxes"),
        ("facial_landmarks", "Facial landmark localization"),
        ("face_tracking", "Face tracking IDs"),
        ("multiple_faces", "Multiple face detection"),
        ("alignment_normalization", "Alignment and normalization"),
        ("emotion_recognition", "Emotion recognition"),
        ("demographic_analysis", "Demographic analysis"),
        ("liveness_detection", "Liveness detection"),
    ]

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
        recent_attendance=recent_attendance,
        student_profile=user,
        recent_students=[],
        capability_status=capability_status,
        capability_items=capability_items,
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


@app.route("/test_detection", methods=["GET"])
def test_detection():
    if not session.get("user") or session.get("role") != "admin":
        return jsonify({"error": "Admin only"}), 403

    result = {
        "cascade_loaded": face_cascade.empty() == False,
        "lbph_support": has_lbph_support(),
        "facemark_support": has_facemark_support(),
        "landmark_model_found": os.path.exists(LBF_LANDMARK_MODEL),
        "demographics_models_found": all(
            os.path.exists(path)
            for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
        ),
        "cascade_path": CASCADE_PATH,
    }

    recognizer, label_map = load_face_recognizer()
    result["recognizer_trained"] = recognizer is not None
    result["labeled_students"] = len(label_map)
    result["student_list"] = [
        {"label": k, "student_id": v[0], "name": v[1]} for k, v in label_map.items()
    ]

    video, camera_label = open_camera([("Default camera (0)", 0)])
    result["camera_accessible"] = video is not None
    result["camera_label"] = camera_label
    if video:
        video.release()

    for idx in [1, 2, 3]:
        v = cv2.VideoCapture(idx)
        if v.isOpened():
            result[f"camera_{idx}_available"] = True
            v.release()
        else:
            result[f"camera_{idx}_available"] = False

    return jsonify(result)


@app.route("/scanner")
def scanner():
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can access the scanner.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

    cloud_hosted = os.environ.get("RENDER", "").lower() == "true"
    capability_status = {
        "real_time_boxes": True,
        "facial_landmarks": has_facemark_support() and os.path.exists(LBF_LANDMARK_MODEL),
        "face_tracking": True,
        "multiple_faces": True,
        "alignment_normalization": True,
        "emotion_recognition": True,
        "demographic_analysis": all(
            os.path.exists(path) for path in [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
        ),
        "liveness_detection": True,
    }
    capability_items = [
        ("real_time_boxes", "Real-time bounding boxes"),
        ("facial_landmarks", "Facial landmark localization"),
        ("face_tracking", "Face tracking IDs"),
        ("multiple_faces", "Multiple face detection"),
        ("alignment_normalization", "Alignment and normalization"),
        ("emotion_recognition", "Emotion recognition"),
        ("demographic_analysis", "Demographic analysis"),
        ("liveness_detection", "Liveness detection"),
    ]
    return render_template(
        "scanner.html",
        cloud_hosted=cloud_hosted,
        last_camera_source=session.get("last_camera_source", "default"),
        last_camera_url=session.get("last_camera_url", ""),
        last_camera_indexes=session.get("last_camera_indexes", "0,1,2"),
        capability_status=capability_status,
        capability_items=capability_items,
    )


@app.route("/start_scan", methods=["POST"])
def start_scan():
    if not session.get("user") or session.get("role") != "admin":
        flash("Only admins can start scanning.", "danger")
        return redirect(url_for("user_dashboard" if session.get("user") else "login"))

    if os.environ.get("RENDER", "").lower() == "true":
        flash("Webcam-based scanning is not available on Render. Run the scanner locally.", "warning")
        return redirect(url_for("scanner"))

    camera_source = request.form.get("camera_source", "default").strip().lower()
    camera_url = request.form.get("camera_url", "").strip()
    camera_indexes_raw = request.form.get("camera_indexes", "0,1,2").strip()

    session["last_camera_source"] = camera_source
    session["last_camera_url"] = camera_url
    session["last_camera_indexes"] = camera_indexes_raw

    camera_candidates = build_camera_candidates(camera_source, camera_url, camera_indexes_raw)
    if not camera_candidates:
        flash("Please provide a phone camera URL before starting scan.", "warning")
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
        logger.error("Recognizer is None - no students loaded")
        return redirect(url_for("scanner"))

    facemark = load_facemark_model()
    age_net, gender_net = load_demographics_models()
    tracker = SimpleFaceTracker()
    track_states = {}

    logger.info(f"Recognizer trained with {len(label_map)} students: {[(k, v[0], v[1]) for k, v in label_map.items()]}")

    video, selected_camera_label = open_camera(camera_candidates)
    if video is None:
        flash(
            "Unable to access selected camera source. If using Iriun/DroidCam, start the app first and try camera index 1 or 2.",
            "danger",
        )
        return redirect(url_for("scanner"))

    flash(f"Scanning started using: {selected_camera_label}.", "info")

    scanned_students = set()
    duplicate_notified = set()
    recognized_faces = 0
    start_time = datetime.now()
    scan_duration = 18

    while (datetime.now() - start_time).seconds < scan_duration:
        grabbed, frame = video.read()
        if not grabbed:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        faces = faces[:8]
        face_rects = [tuple(map(int, face)) for face in faces]

        track_ids = tracker.update(face_rects)
        landmarks_by_index = detect_landmarks(gray_frame, face_rects, facemark)

        if len(face_rects) > 0:
            logger.debug(f"Detected {len(face_rects)} face(s) in frame")

        for idx, (x, y, w, h) in enumerate(face_rects):
            track_id = track_ids[idx] if idx < len(track_ids) else None
            track_state = track_states.setdefault(track_id, {}) if track_id is not None else {}
            landmarks = landmarks_by_index.get(idx)
            face_region_resized = align_and_normalize_face(gray_frame, (x, y, w, h), landmarks)

            try:
                label, confidence = recognizer.predict(face_region_resized)
                logger.debug(f"Prediction: label={label}, confidence={confidence}")
            except cv2.error as e:
                logger.warning(f"Error during face prediction: {e}")
                continue

            emotion_label, emotion_score = estimate_emotion(face_region_resized)
            liveness_label, liveness_score = estimate_liveness(track_state, face_region_resized, landmarks)

            color_face = frame[max(0, y): min(frame.shape[0], y + h), max(0, x): min(frame.shape[1], x + w)]
            age_group, gender = estimate_demographics(color_face, age_net, gender_net) if color_face.size else ("Unknown", "Unknown")

            if confidence < 85 and liveness_label == "Live":
                student_id, name = label_map.get(label, (None, None))
                logger.info(f"✓ MATCH ACCEPTED: label={label}, name={name}, student_id={student_id}, confidence={confidence:.2f}")
                if student_id and student_id not in scanned_students:
                    scanned_students.add(student_id)
                    if mark_attendance(student_id, name):
                        recognized_faces += 1
                        logger.info(f"Attendance marked: {name}")
                    else:
                        logger.warning(f"Duplicate attendance: {name}")
                        if student_id not in duplicate_notified:
                            duplicate_notified.add(student_id)
                display_name = name if name else "Unknown"
            else:
                display_name = "Unknown"
                logger.debug(
                    f"✗ REJECTED: confidence={confidence:.2f}, liveness={liveness_label} ({liveness_score:.2f})"
                )

            box_color = (0, 200, 0) if liveness_label == "Live" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            if landmarks is not None:
                for (lx, ly) in landmarks.astype(int):
                    cv2.circle(frame, (lx, ly), 1, (255, 215, 0), -1)

            track_text = f"Track #{track_id}" if track_id is not None else "Track #?"
            id_text = f"{display_name} ({int(confidence)})"
            emotion_text = f"Emotion: {emotion_label} {emotion_score:.2f}"
            demo_text = f"Age: {age_group} | Gender: {gender}"
            live_text = f"Liveness: {liveness_label} {liveness_score:.2f}"

            cv2.putText(frame, track_text, (x, max(15, y - 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame, id_text, (x, max(30, y - 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            cv2.putText(frame, emotion_text, (x, max(45, y - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, demo_text, (x, y + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, live_text, (x, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        cv2.putText(
            frame,
            f"Faces: {len(face_rects)} | Marked: {recognized_faces} | Press Q to stop",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Attendance Scanner", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Scan interrupted by user (Q pressed)")
            break

    video.release()
    cv2.destroyAllWindows()

    if not scanned_students:
        logger.warning("No faces matched during scan")
        flash("No faces matched during the scan. Try again with better lighting or check test endpoint.", "warning")
    else:
        logger.info(f"Scan complete: {len(scanned_students)} students marked")
        flash(
            f"Scan complete. Marked {recognized_faces} student(s) with multi-face tracking and liveness checks.",
            "success",
        )

    if duplicate_notified:
        flash(
            f"{len(duplicate_notified)} recognized student(s) were already marked for today.",
            "info",
        )

    if facemark is None:
        flash(
            "Landmark model not found. Add models/lbfmodel.yaml to enable full landmark localization.",
            "warning",
        )

    if age_net is None or gender_net is None:
        flash(
            "Demographic model files are missing. Add age/gender Caffe model files in models/ for age/gender prediction.",
            "warning",
        )

    return redirect(url_for("scanner"))


@app.route("/records")
def records():
    if not session.get("user"):
        return redirect(url_for("login"))

    search_date = request.args.get("date", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    search_name = request.args.get("student_name", "").strip()
    search_department = request.args.get("department", "").strip()
    is_admin = session.get("role") == "admin"

    if start_date and end_date and start_date > end_date:
        flash("Start date cannot be later than end date.", "warning")
        return redirect(url_for("records"))

    conn = get_db()

    departments = conn.execute(
        "SELECT DISTINCT department FROM Students WHERE department IS NOT NULL AND department != '' ORDER BY department"
    ).fetchall()

    query = [
        """
        SELECT Attendance.*, COALESCE(Students.department, 'N/A') AS department
        FROM Attendance
        LEFT JOIN Students ON Students.student_id = Attendance.student_id
        """
    ]
    conditions = []
    params = []

    if start_date:
        conditions.append("Attendance.date >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("Attendance.date <= ?")
        params.append(end_date)

    if search_date and not start_date and not end_date:
        conditions.append("Attendance.date = ?")
        params.append(search_date)

    if search_department:
        conditions.append("Students.department = ?")
        params.append(search_department)

    if is_admin and search_name:
        conditions.append("Attendance.name LIKE ?")
        params.append(f"%{search_name}%")

    if not is_admin:
        user = conn.execute("SELECT username, full_name FROM Users WHERE username = ?", (session["user"],)).fetchone()
        student_names = []
        if user:
            if user["full_name"]:
                student_names.append(user["full_name"])
            student_names.append(user["username"])
        student_names = list(dict.fromkeys(student_names))
        if student_names:
            placeholders = ",".join("?" * len(student_names))
            conditions.append(f"Attendance.name IN ({placeholders})")
            params.extend(student_names)
        else:
            conditions.append("1 = 0")

    if conditions:
        query.append("WHERE " + " AND ".join(conditions))

    query.append("ORDER BY Attendance.id DESC")
    records = conn.execute(" ".join(query), params).fetchall()

    total_entries = len(records)
    unique_students = len({row["student_id"] for row in records})
    present_entries = sum(1 for row in records if str(row["status"]).lower() == "present")

    active_filters = {
        "search_date": search_date,
        "start_date": start_date,
        "end_date": end_date,
        "search_name": search_name,
        "search_department": search_department,
    }

    conn.close()
    return render_template(
        "records.html",
        records=records,
        total_entries=total_entries,
        unique_students=unique_students,
        present_entries=present_entries,
        active_filters=active_filters,
        search_date=search_date,
        start_date=start_date,
        end_date=end_date,
        search_name=search_name,
        search_department=search_department,
        departments=departments,
        is_admin=is_admin,
    )


@app.route("/download_csv")
def download_csv():
    if not session.get("user"):
        return redirect(url_for("login"))

    search_date = request.args.get("date", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    search_name = request.args.get("student_name", "").strip()
    search_department = request.args.get("department", "").strip()
    is_admin = session.get("role") == "admin"

    if start_date and end_date and start_date > end_date:
        flash("Start date cannot be later than end date.", "warning")
        return redirect(url_for("records"))

    conn = get_db()

    query = [
        """
        SELECT Attendance.*, COALESCE(Students.department, 'N/A') AS department
        FROM Attendance
        LEFT JOIN Students ON Students.student_id = Attendance.student_id
        """
    ]
    conditions = []
    params = []

    if start_date:
        conditions.append("Attendance.date >= ?")
        params.append(start_date)

    if end_date:
        conditions.append("Attendance.date <= ?")
        params.append(end_date)

    if search_date and not start_date and not end_date:
        conditions.append("Attendance.date = ?")
        params.append(search_date)

    if search_department:
        conditions.append("Students.department = ?")
        params.append(search_department)

    if is_admin and search_name:
        conditions.append("Attendance.name LIKE ?")
        params.append(f"%{search_name}%")

    if not is_admin:
        user = conn.execute("SELECT username, full_name FROM Users WHERE username = ?", (session["user"],)).fetchone()
        student_names = []
        if user:
            if user["full_name"]:
                student_names.append(user["full_name"])
            student_names.append(user["username"])
        student_names = list(dict.fromkeys(student_names))
        if student_names:
            placeholders = ",".join("?" * len(student_names))
            conditions.append(f"Attendance.name IN ({placeholders})")
            params.extend(student_names)
        else:
            conditions.append("1 = 0")

    if conditions:
        query.append("WHERE " + " AND ".join(conditions))

    query.append("ORDER BY Attendance.id DESC")
    records = conn.execute(" ".join(query), params).fetchall()
    conn.close()

    if not records:
        flash("No attendance data available to download.", "warning")
        return redirect(url_for("records"))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Student ID", "Name", "Department", "Date", "Time", "Status"])
    for row in records:
        writer.writerow([row["student_id"], row["name"], row["department"], row["date"], row["time"], row["status"]])

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
