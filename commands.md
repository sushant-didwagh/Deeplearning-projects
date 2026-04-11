# AutoVision Startup Commands

To run the full AutoVision stack, you need to open three separate terminal windows. Follow the instructions below for each terminal.

### Terminal 1: Start the Node.js API Proxy
This server bridges the React frontend and the Python AI backend.
```powershell
# 1. Navigate to the project root folder
cd C:\Users\susha\Downloads\Genai_project

# 2. Start the backend server (runs on port 5000)
npm run dev
```

---

### Terminal 2: Start the React Frontend
This is the client dashboard of the application where you upload images.
```powershell
# 1. Navigate to the client folder
cd C:\Users\susha\Downloads\Genai_project\client

# 2. Start the Vite development server (runs on port 5173)
npm run dev
```

---

### Terminal 3: Start the Python YOLOv8 API
This FastAPI server runs the actual YOLO AI model to detect objects.
```powershell
# 1. Navigate to the project root folder
cd C:\Users\susha\Downloads\Genai_project

# 2. Activate the Python virtual environment
.\.venv\Scripts\activate

#install
python -m pip install --upgrade ultralytics

# 3. Start the Python server (runs on port 8000)
python main.py
```
