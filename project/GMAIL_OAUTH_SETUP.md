# Gmail OAuth Setup Guide

## How to Enable Gmail Login

### Step 1: Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or select existing one)
3. Enable **Google+ API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Select **Web application**
6. Add authorized redirect URI:
   ```
   http://127.0.0.1:5000/auth/google/callback
   ```
   Add your Render domain too, for example:
   ```
   https://your-app-name.onrender.com/auth/google/callback
   ```
7. Copy the **Client ID** and **Client Secret**

### Step 2: Set Environment Variables

**On Windows PowerShell:**

```powershell
$env:GOOGLE_CLIENT_ID = "your-client-id-here"
$env:GOOGLE_CLIENT_SECRET = "your-client-secret-here"
$env:GOOGLE_REDIRECT_URI = "http://127.0.0.1:5000/auth/google/callback"
```

**Or create a `.env` file in the project folder:**

```
GOOGLE_CLIENT_ID=your-client-id-here
GOOGLE_CLIENT_SECRET=your-client-secret-here
GOOGLE_REDIRECT_URI=http://127.0.0.1:5000/auth/google/callback
```

For Render, set `GOOGLE_REDIRECT_URI` to your public Render callback URL.

### Step 3: Restart the Flask Server

```powershell
.\.venv\Scripts\python.exe app.py
```

### Step 4: Login with Gmail

- Visit `http://127.0.0.1:5000/login`
- Click **"Sign in with Gmail"**
- Users will be automatically registered as regular users (role: `user`)
- They can view attendance records but cannot register students or access the scanner

### Render Notes

- Set `GOOGLE_CLIENT_ID`
- Set `GOOGLE_CLIENT_SECRET`
- Set `GOOGLE_REDIRECT_URI` to your Render callback URL
- The callback URL must match the one added in Google Cloud Console exactly

## Features

✅ Gmail OAuth Login  
✅ Automatic user account creation  
✅ Role-based access control (admin/user)  
✅ Email-based username  

## Troubleshooting

- If Gmail login doesn't work, check that `GOOGLE_CLIENT_ID` is set
- Redirect URI must exactly match what you configured in Google Cloud Console
- Use `http://127.0.0.1:5000` for local development, not `localhost`
