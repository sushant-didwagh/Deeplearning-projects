import { useState, useRef, useCallback } from "react";
import "./App.css";

// ── API endpoint (Express proxy)
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

// ── Detection confidence color helper
const confColor = (c) =>
  c >= 0.8 ? "#00C896" : c >= 0.6 ? "#F0A500" : "#FF6B6B";

// ── Class icon map
const classIcon = {
  person: "🚶",
  car: "🚗",
  truck: "🚛",
  bus: "🚌",
  bicycle: "🚲",
  motorcycle: "🏍️",
  "traffic light": "🚦",
  "stop sign": "🛑",
  train: "🚆",
  default: "📦",
};

export default function App() {
  const [view, setView] = useState("landing"); // "landing" | "detect"
  const [image, setImage] = useState(null);    // { file, previewUrl }
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [confidence, setConfidence] = useState(0.35);
  const [dragOver, setDragOver] = useState(false);

  const fileInputRef = useRef();

  // ── Handle file selection
  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) {
      setError("Please upload a valid image (JPEG, PNG, WEBP, BMP).");
      return;
    }
    setError("");
    setResult(null);
    const previewUrl = URL.createObjectURL(file);
    setImage({ file, previewUrl });
  }, []);

  const onFileInput = (e) => handleFile(e.target.files[0]);
  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

  // ── Run detection
  const detect = async () => {
    if (!image) return;
    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", image.file);
    form.append("confidence", confidence);
    form.append("iou_threshold", 0.45);

    try {
      const res = await fetch(`${API_BASE}/detect`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || err.error || "Detection failed.");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null);
    setResult(null);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ════════════════════════════════════════════
  // LANDING PAGE VIEW
  // ════════════════════════════════════════════
  if (view === "landing") {
    return (
      <div className="landing">
        {/* Nav */}
        <nav className="nav">
          <div className="nav-logo">
            <span className="logo-icon">⬡</span>
            <span className="logo-text">AutoVision</span>
          </div>
          <div className="nav-links">
            <a href="#features">Features</a>
            <a href="#model">Model</a>
            <button className="btn-nav" onClick={() => setView("detect")}>
              Launch App →
            </button>
          </div>
        </nav>

        {/* Hero */}
        <section className="hero">
          <div className="hero-badge">YOLOv8x · Real-time Detection</div>
          <h1 className="hero-title">
            See the Road<br />
            <span className="accent">Like an AI Does</span>
          </h1>
          <p className="hero-sub">
            Enterprise-grade object detection for autonomous vehicles — cars, pedestrians,
            trucks, traffic lights and more. Powered by the state-of-the-art YOLOv8x model.
          </p>
          <div className="hero-cta">
            <button className="btn-primary" onClick={() => setView("detect")}>
              Try Detection Free →
            </button>
            <a className="btn-ghost" href="#model">Learn about the model</a>
          </div>

          {/* Mock dashboard preview */}
          <div className="hero-preview">
            <div className="preview-bar">
              <span className="dot red" />
              <span className="dot amber" />
              <span className="dot green" />
              <span className="preview-title">AutoVision · Live Feed</span>
            </div>
            <div className="preview-body">
              <div className="preview-boxes">
                <div className="mock-box car">Car 97%</div>
                <div className="mock-box person">Person 92%</div>
                <div className="mock-box truck">Truck 88%</div>
                <div className="mock-box light">Traffic Light 95%</div>
              </div>
              <div className="preview-stats">
                <div className="stat-chip">🚗 3 Cars</div>
                <div className="stat-chip">🚶 2 Persons</div>
                <div className="stat-chip">🚛 1 Truck</div>
                <div className="stat-chip">🚦 2 Lights</div>
              </div>
            </div>
          </div>
        </section>

        {/* Features */}
        <section className="features" id="features">
          <h2 className="section-title">Built for Autonomous Systems</h2>
          <div className="features-grid">
            {[
              { icon: "⚡", title: "YOLOv8x Model", desc: "Largest, most accurate YOLO architecture — 68.7% mAP on COCO benchmark. Best accuracy for production AV use cases." },
              { icon: "🎯", title: "80+ Object Classes", desc: "Detects cars, trucks, buses, motorcycles, pedestrians, cyclists, traffic lights, stop signs, and more." },
              { icon: "📐", title: "Bounding Box Output", desc: "Returns pixel coordinates, normalized coords, confidence scores, and class labels for every detected object." },
              { icon: "🔌", title: "REST API Ready", desc: "FastAPI backend with JSON responses. Plug directly into your MERN stack or any other application." },
              { icon: "🛡️", title: "AV-Optimized", desc: "Filters and highlights autonomous vehicle-relevant classes. Prioritizes road safety critical objects." },
              { icon: "📸", title: "Annotated Output", desc: "Returns the original image with colored bounding boxes drawn, as a base64-encoded JPEG." },
            ].map((f) => (
              <div className="feature-card" key={f.title}>
                <div className="feature-icon">{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Model Info */}
        <section className="model-section" id="model">
          <div className="model-content">
            <div className="model-text">
              <h2>YOLOv8x — The Best Pretrained Choice</h2>
              <p>
                YOLOv8x (extra-large) by Ultralytics is the current state-of-the-art
                single-stage object detector. It achieves <strong>68.7% mAP</strong> on the
                COCO val2017 dataset, making it ideal for safety-critical autonomous vehicle
                perception tasks where accuracy matters most.
              </p>
              <ul className="model-specs">
                <li><span>Parameters</span><strong>68.2M</strong></li>
                <li><span>mAP50-95 (COCO)</span><strong>53.9</strong></li>
                <li><span>Input Size</span><strong>640×640</strong></li>
                <li><span>Training Data</span><strong>COCO 2017</strong></li>
                <li><span>Classes</span><strong>80</strong></li>
                <li><span>Framework</span><strong>Ultralytics / PyTorch</strong></li>
              </ul>
            </div>
            <div className="model-comparison">
              <h3>Model Accuracy Comparison</h3>
              {[
                { name: "YOLOv8n", map: 37.3, pct: 54, note: "Fastest" },
                { name: "YOLOv8s", map: 44.9, pct: 66, note: "" },
                { name: "YOLOv8m", map: 50.2, pct: 74, note: "" },
                { name: "YOLOv8l", map: 52.9, pct: 78, note: "" },
                { name: "YOLOv8x", map: 53.9, pct: 100, note: "← We use this" },
              ].map((m) => (
                <div className="bar-row" key={m.name}>
                  <span className="bar-label">{m.name}</span>
                  <div className="bar-track">
                    <div
                      className={`bar-fill ${m.name === "YOLOv8x" ? "active" : ""}`}
                      style={{ width: `${m.pct}%` }}
                    />
                  </div>
                  <span className="bar-val">{m.map}</span>
                  {m.note && <span className="bar-note">{m.note}</span>}
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="cta-section">
          <h2>Ready to detect?</h2>
          <p>Upload any road scene image and see YOLOv8x in action.</p>
          <button className="btn-primary large" onClick={() => setView("detect")}>
            Open Detection App →
          </button>
        </section>

        <footer className="footer">
          <p>AutoVision · YOLOv8x Object Detection · MERN Stack Integration</p>
        </footer>
      </div>
    );
  }

  // ════════════════════════════════════════════
  // DETECTION APP VIEW
  // ════════════════════════════════════════════
  return (
    <div className="app">
      {/* Topbar */}
      <div className="app-topbar">
        <button className="back-btn" onClick={() => { reset(); setView("landing"); }}>
          ← AutoVision
        </button>
        <span className="app-topbar-title">Object Detection</span>
        <span className="model-badge">YOLOv8x</span>
      </div>

      <div className="app-body">
        {/* Left panel — upload + controls */}
        <div className="panel-left">
          <h2 className="panel-title">Upload Image</h2>

          {/* Drop zone */}
          <div
            className={`dropzone ${dragOver ? "drag-active" : ""} ${image ? "has-image" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
          >
            {image ? (
              <img src={image.previewUrl} alt="preview" className="preview-img" />
            ) : (
              <div className="dropzone-inner">
                <div className="drop-icon">📷</div>
                <p>Drag & drop an image here</p>
                <span>or click to browse</span>
                <small>JPEG · PNG · WEBP · BMP · Max 20MB</small>
              </div>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={onFileInput}
            />
          </div>

          {/* Confidence slider */}
          <div className="control-group">
            <label>
              Confidence Threshold
              <strong>{Math.round(confidence * 100)}%</strong>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="slider"
            />
            <div className="slider-hints">
              <span>More detections</span>
              <span>Higher precision</span>
            </div>
          </div>

          {/* Actions */}
          <div className="action-row">
            <button
              className="btn-detect"
              onClick={detect}
              disabled={!image || loading}
            >
              {loading ? "Detecting…" : "⬡ Run Detection"}
            </button>
            {(image || result) && (
              <button className="btn-reset" onClick={reset}>Reset</button>
            )}
          </div>

          {error && <div className="error-box">⚠ {error}</div>}

          {/* Stats strip */}
          {result && (
            <div className="stats-strip">
              <div className="stat">
                <span>{result.total_detections}</span>
                <label>Total Objects</label>
              </div>
              <div className="stat">
                <span>{result.av_relevant_count}</span>
                <label>AV Relevant</label>
              </div>
              <div className="stat">
                <span>{result.inference_time_ms}ms</span>
                <label>Inference</label>
              </div>
              <div className="stat">
                <span>{Object.keys(result.class_counts).length}</span>
                <label>Classes Found</label>
              </div>
            </div>
          )}
        </div>

        {/* Right panel — results */}
        <div className="panel-right">
          {/* Output image */}
          <div className="output-image-wrap">
            {result ? (
              <>
                <img
                  src={result.annotated_image}
                  alt="annotated"
                  className="output-img"
                />
                <a
                  href={result.annotated_image}
                  download="detection_result.jpg"
                  className="download-btn"
                >
                  ↓ Download
                </a>
              </>
            ) : (
              <div className="output-placeholder">
                {loading ? (
                  <div className="loader-wrap">
                    <div className="spinner" />
                    <p>Running YOLOv8x inference…</p>
                  </div>
                ) : (
                  <p className="placeholder-text">
                    Annotated image will appear here after detection
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Detection table */}
          {result && result.detections.length > 0 && (
            <div className="detections-panel">
              <h3 className="detections-title">
                Detected Objects
                <span className="count-badge">{result.total_detections}</span>
              </h3>

              {/* Class summary pills */}
              <div className="class-pills">
                {Object.entries(result.class_counts).map(([cls, count]) => (
                  <span key={cls} className="class-pill">
                    {classIcon[cls] || classIcon.default} {cls} ({count})
                  </span>
                ))}
              </div>

              {/* Detection list */}
              <div className="det-list">
                {result.detections.map((det, i) => (
                  <div className="det-item" key={i}>
                    <div className="det-left">
                      <span className="det-icon">
                        {classIcon[det.class_name] || classIcon.default}
                      </span>
                      <div>
                        <strong className="det-name">{det.class_name}</strong>
                        {det.is_av_relevant && (
                          <span className="av-badge">AV</span>
                        )}
                        <div className="det-meta">
                          [{det.bbox.join(", ")}] · {det.area_px.toLocaleString()}px²
                        </div>
                      </div>
                    </div>
                    <div className="det-conf-wrap">
                      <div
                        className="det-conf-bar"
                        style={{
                          width: `${Math.round(det.confidence * 100)}%`,
                          background: confColor(det.confidence),
                        }}
                      />
                      <span
                        className="det-conf-label"
                        style={{ color: confColor(det.confidence) }}
                      >
                        {Math.round(det.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result && result.detections.length === 0 && (
            <div className="no-detections">
              No objects detected above {Math.round(confidence * 100)}% confidence.
              Try lowering the threshold.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
