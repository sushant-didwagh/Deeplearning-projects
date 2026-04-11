A application where upload image of traffic/lanes/cars and get result in form of detected image.
# AutoVision — YOLOv8 Object Detection for Autonomous Vehicles
## MERN Stack Integration

---

## Architecture

```
React Frontend (port 3000)
       ↓  POST /api/detect (multipart image)
Express.js Node Server (port 5000)
       ↓  POST /detect (forwards to Python)
FastAPI + YOLOv8x Python Server (port 8000)
       ↓  Returns JSON with detections + annotated image (base64)
```

---

## 1. Python FastAPI Backend (YOLOv8x)

### Setup
```bash
cd backend_python
pip install -r requirements.txt
python main.py
```

On first run, YOLOv8x weights (~130 MB) are auto-downloaded from Ultralytics.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/detect` | Detect objects in image |

### POST /detect — Request
- **Content-Type**: `multipart/form-data`
- **file**: image file (JPEG / PNG / WEBP / BMP, max 20MB)
- **confidence** (optional): float 0.1–0.95, default `0.35`
- **iou_threshold** (optional): float, default `0.45`

### POST /detect — Response
```json
{
  "success": true,
  "model": "YOLOv8x",
  "inference_time_ms": 245.3,
  "image_size": { "width": 1280, "height": 720 },
  "total_detections": 7,
  "av_relevant_count": 5,
  "class_counts": { "car": 3, "person": 2, "truck": 1, "traffic light": 1 },
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.9342,
      "bbox": [120, 300, 480, 560],
      "bbox_normalized": [0.094, 0.417, 0.375, 0.778],
      "area_px": 115200,
      "color": "#45B7D1",
      "is_av_relevant": true
    }
  ],
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

### GPU Acceleration
In `main.py`, change `device="cpu"` to `device="cuda"` (requires CUDA + PyTorch GPU build).

---

## 2. Node.js / Express Backend (Proxy)

```bash
cd backend_node
npm install
npm run dev        # development with nodemon
npm start          # production
```

**Environment variables:**
```
PORT=5000
PYTHON_API_URL=http://localhost:8000
NODE_ENV=production   # to serve React build
```

---

## 3. React Frontend

```bash
cd frontend_react
npx create-react-app .     # if not already initialized
# OR: npm create vite@latest . -- --template react

# Copy App.jsx and App.css into src/

npm install
npm start
```

**Environment variables (`.env`):**
```
REACT_APP_API_URL=http://localhost:5000/api
```

---

## Model Choice: Why YOLOv8x?

| Model | mAP50-95 | Speed | Use case |
|-------|----------|-------|----------|
| YOLOv8n | 37.3 | Fastest | Edge devices |
| YOLOv8s | 44.9 | Fast | Mobile |
| YOLOv8m | 50.2 | Medium | Balanced |
| YOLOv8l | 52.9 | Slower | High accuracy |
| **YOLOv8x** | **53.9** | Slowest | **Best accuracy ← we use** |

For real-time video on embedded hardware (Jetson, etc.), switch to `yolov8n.pt` or `yolov8s.pt`.

---

## Detected Object Classes (AV-Relevant)

- 🚗 car · 🚛 truck · 🚌 bus · 🚂 train
- 🚶 person · 🚲 bicycle · 🏍️ motorcycle
- 🚦 traffic light · 🛑 stop sign · 🅿️ parking meter

---

## Production Deployment

1. Build React: `npm run build` in `frontend_react/`
2. Set `NODE_ENV=production` in Express — it will serve the React build
3. Run Python API behind gunicorn: `gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker`
4. Use nginx as reverse proxy in front of Express
