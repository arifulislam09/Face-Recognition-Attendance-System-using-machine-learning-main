# Face Recognition Attendance System

This is a beginner-friendly Python Flask attendance system using OpenCV and face_recognition.
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

1. Open VS Code and open the `project` folder.
2. Open a terminal in VS Code.
3. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   ```
4. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate
   ```
5. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
6. Run the Flask app:
   ```powershell
   python app.py
   ```
7. Open your browser to:
   ```text
   http://127.0.0.1:5000
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
