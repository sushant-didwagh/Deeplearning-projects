/**
 * AutoVision — Express.js Proxy Server
 * Acts as the Node/Express layer between React frontend and Python FastAPI
 * Run: node server.js  (after: npm install)
 */

const express = require("express");
const cors = require("cors");
const multer = require("multer");
const FormData = require("form-data");
const fetch = require("node-fetch");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 5000;
const PYTHON_API = process.env.PYTHON_API_URL || "http://localhost:8000";

// ── Middleware
app.use(cors()); // Allow all origins to prevent Vite port switching issues
app.use(express.json());

// Multer — store in memory (max 20 MB)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/bmp", "image/webp"];
    if (allowed.includes(file.mimetype)) cb(null, true);
    else cb(new Error("Invalid file type. Use JPEG, PNG, BMP, or WEBP."));
  },
});

// ── Routes

// Health check
app.get("/api/health", async (req, res) => {
  try {
    const r = await fetch(`${PYTHON_API}/health`);
    const data = await r.json();
    res.json({ node: "ok", python: data });
  } catch {
    res.status(503).json({ node: "ok", python: "unreachable" });
  }
});

// Single image detection
app.post("/api/detect", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image file provided." });

  const confidence = parseFloat(req.body.confidence) || 0.35;
  const iou = parseFloat(req.body.iou_threshold) || 0.45;

  const form = new FormData();
  form.append("file", req.file.buffer, {
    filename: req.file.originalname,
    contentType: req.file.mimetype,
  });

  try {
    const response = await fetch(
      `${PYTHON_API}/detect?confidence=${confidence}&iou_threshold=${iou}`,
      { method: "POST", body: form, headers: form.getHeaders() }
    );

    if (!response.ok) {
      const err = await response.json();
      return res.status(response.status).json(err);
    }

    const result = await response.json();
    res.json(result);
  } catch (err) {
    console.error("Python API error:", err.message);
    res.status(502).json({ error: "Detection service unavailable.", detail: err.message });
  }
});

// Serve React build in production
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "client/build")));
  app.get("*", (req, res) =>
    res.sendFile(path.join(__dirname, "client/build", "index.html"))
  );
}

// Error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`\n🚗 AutoVision Node server running on http://localhost:${PORT}`);
  console.log(`   Python API target: ${PYTHON_API}\n`);
});
