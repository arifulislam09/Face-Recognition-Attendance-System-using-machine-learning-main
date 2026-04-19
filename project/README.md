# Face Recognition Attendance System

This is a beginner-friendly Python Flask attendance system using OpenCV LBPH face recognition.
The app stores student data in SQLite, captures faces from the webcam, and automatically marks attendance.

## Folder Structure

project/
│── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── images/
│── templates/
│   ├── login.html
│   ├── dashboard.html
│   ├── register.html
│   ├── scanner.html
│   ├── records.html
│── dataset/
│── app.py
│── database.db
│── requirements.txt

## Setup in VS Code

> Important: This project requires Python 3.11 on Windows.

1. **Install Python 3.11**:
   - Download from: https://www.python.org/downloads/release/python-3110/
   - Install it (add to PATH if asked).

2. Open VS Code and open the `project` folder.

3. Open a terminal in VS Code (inside `project` folder).

4. Run the setup script:
   ```powershell
   .\setup.bat
   ```
   This will create venv, activate it, and install dependencies.

5. Run the Flask app:
   - From terminal: `python app.py`
   - Or from VS Code: Press F5 (uses launch.json)

6. Open your browser to:
   ```text
   http://127.0.0.1:5000
   ```

## Manual Setup (if batch file doesn't work)

1. Install Python 3.11 as above.

2. In terminal (inside `project` folder):
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   python app.py
   ```


## Login Details

- Username: `admin`
- Password: `admin123`

## Usage

- Register students with a face photo.
- Go to Attendance Scanner and press Start Scan.
- Attendance is matched automatically with registered faces.
- View all entries in Attendance Records.
- Download attendance as a CSV file.

## Notes

- Use a clear webcam image for better face matching.
- The app prevents duplicate attendance for the same student on the same day.
- Images are saved under `static/images`.
