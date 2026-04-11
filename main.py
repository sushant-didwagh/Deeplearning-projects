"""
YOLOv8 Object Detection API for Autonomous Vehicle
FastAPI backend — handles image uploads and returns detected objects
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import base64
import time
from PIL import Image
import numpy as np

# ── Install: pip install fastapi uvicorn ultralytics pillow numpy python-multipart
from ultralytics import YOLO

app = FastAPI(
    title="AutoVision Object Detection API",
    description="YOLOv8-based object detection optimized for autonomous vehicle perception",
    version="1.0.0"
)

# Allow requests from your React frontend (update origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load YOLOv8x — largest/most accurate pretrained model (COCO-trained)
# yolov8n = nano (fastest), yolov8s = small, yolov8m = medium,
# yolov8l = large, yolov8x = extra-large (most accurate) ← we use this
# First run auto-downloads ~130 MB weights from Ultralytics
model = YOLO("yolov8x.pt")

# Autonomous vehicle relevant classes from COCO dataset
AV_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    6:  "train",
    7:  "truck",
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    56: "chair",
    60: "dining table",
}

# Color map for bounding box rendering (hex per class)
CLASS_COLORS = {
    "person":        "#FF6B6B",
    "bicycle":       "#4ECDC4",
    "car":           "#45B7D1",
    "motorcycle":    "#96CEB4",
    "bus":           "#FFEAA7",
    "train":         "#DDA0DD",
    "truck":         "#F0A500",
    "traffic light": "#00FF7F",
    "fire hydrant":  "#FF4757",
    "stop sign":     "#FF6348",
    "parking meter": "#A29BFE",
    "bench":         "#FD79A8",
}


def draw_boxes_on_image(image: Image.Image, detections: list) -> str:
    """Draw bounding boxes and return base64 encoded result image."""
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)

    for det in detections:
        box = det["bbox"]            # [x1, y1, x2, y2]
        label = det["class_name"]
        conf = det["confidence"]
        color = det.get("color", "#FF0000")

        # Draw rectangle
        draw.rectangle(box, outline=color, width=3)

        # Draw label background
        text = f"{label} {conf:.0%}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        bbox_text = draw.textbbox((box[0], box[1] - 22), text, font=font)
        draw.rectangle(bbox_text, fill=color)
        draw.text((box[0], box[1] - 22), text, fill="white", font=font)

    # Encode to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
async def root():
    return {"message": "AutoVision Object Detection API", "model": "YOLOv8x", "status": "ready"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: float = 0.35,
    iou_threshold: float = 0.45,
):
    """
    Detect objects in an uploaded image using YOLOv8x.

    Args:
        file: Image file (JPEG, PNG, BMP, WEBP supported)
        confidence: Minimum confidence threshold (default 0.35)
        iou_threshold: IoU threshold for NMS (default 0.45)

    Returns:
        JSON with detections, annotated image (base64), and stats
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, BMP, or WEBP."
        )

    # Read image bytes
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 20MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot open image: {str(e)}")

    img_width, img_height = image.size

    # ── Run YOLOv8 inference
    start_time = time.time()
    results = model.predict(
        source=np.array(image),
        conf=confidence,
        iou=iou_threshold,
        verbose=False,
        device="cpu",   # change to "cuda" if GPU is available
    )
    inference_time = round((time.time() - start_time) * 1000, 1)  # ms

    # ── Parse detections
    detections = []
    class_counts = {}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            class_name = model.names[cls_id]

            # x1, y1, x2, y2 in pixel coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": round(conf_val, 4),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "bbox_normalized": [
                    round(x1 / img_width, 4),
                    round(y1 / img_height, 4),
                    round(x2 / img_width, 4),
                    round(y2 / img_height, 4),
                ],
                "area_px": round((x2 - x1) * (y2 - y1)),
                "color": CLASS_COLORS.get(class_name, "#FFFFFF"),
                "is_av_relevant": cls_id in AV_CLASSES,
            }
            detections.append(detection)

            # Count per class
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # ── Draw boxes and encode output image
    annotated_b64 = draw_boxes_on_image(image.copy(), detections)

    # ── Summary statistics
    av_relevant = [d for d in detections if d["is_av_relevant"]]

    return JSONResponse(content={
        "success": True,
        "model": "YOLOv8x",
        "inference_time_ms": inference_time,
        "image_size": {"width": img_width, "height": img_height},
        "total_detections": len(detections),
        "av_relevant_count": len(av_relevant),
        "class_counts": class_counts,
        "detections": detections,
        "annotated_image": annotated_b64,     # base64 JPEG with bounding boxes
        "thresholds": {
            "confidence": confidence,
            "iou": iou_threshold,
        }
    })


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """Detect objects across multiple images at once (max 5)."""
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Max 5 images per batch request.")

    results_list = []
    for f in files:
        result = await detect_objects(f)
        results_list.append(result.body)

    return JSONResponse(content={"results": results_list})


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
