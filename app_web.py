"""
app_web.py
Proyecto 2 - Medidor de Engagement Facial

Aplicacion web funcional:
- Captura la camara desde el navegador.
- Envia frames al backend Flask.
- Detecta FaceMesh con MediaPipe.
- Calcula features geometricas.
- Predice engagement con el modelo entrenado.
"""

from threading import Lock
from pathlib import Path
import base64
import os
import tempfile

import cv2
from flask import Flask, Response, jsonify, render_template_string, request
import numpy as np
import pandas as pd

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / f"engagement_facial_web_mpl_{os.getpid()}"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import mediapipe as mp

import demo_webcam


app = Flask(__name__)

modelo, scaler, columnas_features = demo_webcam.cargar_recursos_modelo()
face_mesh_lock = Lock()
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

COLOR_HEX = {
    "concentrado": "#22c55e",
    "confundido": "#f97316",
    "aburrido": "#9ca3af",
    "sorprendido": "#fde047",
    "sin rostro": "#ef4444",
}

FAVICON_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="12" fill="#11161c"/>
  <circle cx="32" cy="31" r="18" fill="none" stroke="#22c55e" stroke-width="5"/>
  <circle cx="25" cy="27" r="3" fill="#22c55e"/>
  <circle cx="39" cy="27" r="3" fill="#22c55e"/>
  <path d="M24 39 Q32 45 40 39" fill="none" stroke="#f97316" stroke-width="4" stroke-linecap="round"/>
</svg>
"""

HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EngageFace Aula</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171b21;
      --panel-2: #202631;
      --line: #303846;
      --text: #f7fafc;
      --muted: #a7b0bf;
      --accent: #22c55e;
      --red: #ef4444;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, Helvetica, sans-serif;
    }

    .shell {
      width: min(1240px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 58px;
      margin-bottom: 18px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 50px;
      height: 50px;
      border: 1px solid #415064;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background:
        linear-gradient(135deg, rgba(34, 197, 94, 0.22), rgba(249, 115, 22, 0.16)),
        #11161c;
      flex: 0 0 auto;
    }

    .brand-face {
      width: 27px;
      height: 27px;
      border: 2px solid var(--accent);
      border-radius: 50%;
      position: relative;
    }

    .brand-face::before,
    .brand-face::after {
      content: "";
      position: absolute;
      top: 8px;
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background: var(--accent);
    }

    .brand-face::before { left: 6px; }
    .brand-face::after { right: 6px; }

    .brand-mouth {
      position: absolute;
      left: 7px;
      bottom: 6px;
      width: 11px;
      height: 5px;
      border-bottom: 2px solid #f97316;
      border-radius: 0 0 12px 12px;
    }

    h1 {
      margin: 0;
      font-size: 25px;
      font-weight: 800;
      letter-spacing: 0;
    }

    .subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }

    .status-pill {
      min-width: 230px;
      border: 1px solid var(--line);
      background: #090b0f;
      padding: 10px 14px;
      border-radius: 8px;
      text-align: center;
      font-size: 14px;
      font-weight: 800;
      color: var(--accent);
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 330px;
      gap: 16px;
      align-items: start;
    }

    .stage {
      position: relative;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #050608;
      aspect-ratio: 16 / 9;
    }

    video,
    canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    video { transform: scaleX(-1); }

    .banner {
      position: absolute;
      left: 0;
      top: 0;
      width: 100%;
      height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 22px;
      background: rgba(0, 0, 0, 0.78);
      z-index: 3;
      font-size: 24px;
      font-weight: 900;
      color: var(--accent);
    }

    .banner small {
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
    }

    .side-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      overflow: hidden;
    }

    .panel-section {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }

    .panel-section:last-child { border-bottom: 0; }

    .section-title {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .state-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0b0f14;
      padding: 14px;
    }

    .state-label {
      color: var(--accent);
      font-size: 28px;
      font-weight: 900;
      line-height: 1.05;
    }

    .state-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      min-height: 66px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .metric strong {
      display: block;
      margin-top: 6px;
      color: var(--text);
      font-size: 18px;
      font-weight: 800;
    }

    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      color: var(--text);
      min-height: 40px;
      padding: 0 16px;
      font-size: 14px;
      font-weight: 800;
      cursor: pointer;
    }

    button:hover { border-color: var(--accent); }

    .primary {
      background: var(--accent);
      color: #041007;
      border-color: transparent;
    }

    .hint {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.35;
    }

    .error { color: #fca5a5; }

    .legend {
      display: grid;
      gap: 8px;
    }

    .legend-row {
      display: grid;
      grid-template-columns: 14px 1fr;
      gap: 8px;
      align-items: center;
      min-height: 24px;
      color: var(--text);
      font-size: 14px;
      font-weight: 700;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--dot-color);
    }

    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
    }

    @media (max-width: 720px) {
      .shell {
        width: min(100% - 20px, 1120px);
        padding: 12px 0;
      }

      .topbar { display: block; }
      .brand { margin-bottom: 10px; }
      h1 { font-size: 19px; }
      .status-pill { width: 100%; }

      .banner {
        height: 58px;
        font-size: 18px;
        padding: 0 14px;
      }

      .banner small { display: none; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <div class="brand-face"><span class="brand-mouth"></span></div>
        </div>
        <div>
          <h1>EngageFace Aula</h1>
          <div class="subtitle">Medidor de engagement facial en aulas virtuales</div>
        </div>
      </div>
      <div id="statusPill" class="status-pill">CAMARA DETENIDA</div>
    </header>

    <div class="workspace">
      <section class="stage" aria-label="Camara y deteccion facial">
        <video id="video" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas>
        <div class="banner">
          <span id="bannerText">Engagement: SIN ROSTRO</span>
          <small id="frameState">0 FPS</small>
        </div>
      </section>

      <aside class="side-panel" aria-label="Panel de analitica">
        <section class="panel-section">
          <h2 class="section-title">Estado actual</h2>
          <div class="state-card">
            <div id="stateLabel" class="state-label">SIN ROSTRO</div>
            <div class="state-meta">
              <div class="metric">
                <span>Confianza</span>
                <strong id="confidenceValue">--</strong>
              </div>
              <div class="metric">
                <span>Rostro</span>
                <strong id="faceValue">No</strong>
              </div>
            </div>
          </div>
        </section>

        <section class="panel-section">
          <h2 class="section-title">Controles</h2>
          <div class="controls">
            <button id="startBtn" class="primary" type="button">Iniciar camara</button>
            <button id="stopBtn" type="button">Detener</button>
          </div>
          <p id="hint" class="hint">Sistema listo para iniciar.</p>
        </section>

        <section class="panel-section">
          <h2 class="section-title">Clases</h2>
          <div class="legend">
            <div class="legend-row"><span class="dot" style="--dot-color:#22c55e"></span>Concentrado</div>
            <div class="legend-row"><span class="dot" style="--dot-color:#f97316"></span>Confundido</div>
            <div class="legend-row"><span class="dot" style="--dot-color:#9ca3af"></span>Aburrido</div>
            <div class="legend-row"><span class="dot" style="--dot-color:#fde047"></span>Sorprendido</div>
          </div>
        </section>
      </aside>
    </div>

    <canvas id="capture" style="display:none;"></canvas>
  </main>

  <script>
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const capture = document.getElementById("capture");
    const bannerText = document.getElementById("bannerText");
    const statusPill = document.getElementById("statusPill");
    const stateLabel = document.getElementById("stateLabel");
    const confidenceValue = document.getElementById("confidenceValue");
    const faceValue = document.getElementById("faceValue");
    const frameState = document.getElementById("frameState");
    const hint = document.getElementById("hint");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");

    const overlayCtx = overlay.getContext("2d");
    const captureCtx = capture.getContext("2d");
    const colorMap = {
      "concentrado": "#22c55e",
      "confundido": "#f97316",
      "aburrido": "#9ca3af",
      "sorprendido": "#fde047",
      "sin rostro": "#ef4444"
    };

    let stream = null;
    let timer = null;
    let processing = false;
    let labelHistory = [];
    let frames = 0;
    let lastFpsTime = performance.now();

    function setStatus(label, color) {
      const upper = label.toUpperCase();
      bannerText.textContent = `Engagement: ${upper}`;
      statusPill.textContent = upper;
      stateLabel.textContent = upper;
      bannerText.style.color = color;
      statusPill.style.color = color;
      stateLabel.style.color = color;
      document.documentElement.style.setProperty("--accent", color);
    }

    function setConfidence(value) {
      confidenceValue.textContent = value === null || value === undefined
        ? "--"
        : `${Math.round(value * 100)}%`;
    }

    function tickFps() {
      frames += 1;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        frameState.textContent = `${frames} FPS`;
        frames = 0;
        lastFpsTime = now;
      }
    }

    function smoothLabel(label) {
      if (label === "sin rostro") {
        labelHistory = [];
        return label;
      }

      labelHistory.push(label);
      if (labelHistory.length > 7) labelHistory.shift();

      const counts = {};
      for (const item of labelHistory) counts[item] = (counts[item] || 0) + 1;
      return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
    }

    function resizeCanvas() {
      const rect = overlay.getBoundingClientRect();
      overlay.width = Math.round(rect.width);
      overlay.height = Math.round(rect.height);
    }

    function drawLandmarks(points, sourceWidth, sourceHeight) {
      resizeCanvas();
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      overlayCtx.fillStyle = "rgba(245, 245, 245, 0.95)";
      overlayCtx.strokeStyle = "rgba(80, 220, 220, 0.8)";
      overlayCtx.lineWidth = 1;

      const scaleX = overlay.width / sourceWidth;
      const scaleY = overlay.height / sourceHeight;
      const mapped = points.map((point) => ({
        x: overlay.width - point.x * scaleX,
        y: point.y * scaleY
      }));

      const lines = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [14, 15]];
      for (const [a, b] of lines) {
        if (!mapped[a] || !mapped[b]) continue;
        overlayCtx.beginPath();
        overlayCtx.moveTo(mapped[a].x, mapped[a].y);
        overlayCtx.lineTo(mapped[b].x, mapped[b].y);
        overlayCtx.stroke();
      }

      for (const point of mapped) {
        overlayCtx.beginPath();
        overlayCtx.arc(point.x, point.y, 2.4, 0, Math.PI * 2);
        overlayCtx.fill();
      }
    }

    async function sendFrame() {
      if (!stream || processing || video.readyState < 2) return;
      processing = true;

      const width = video.videoWidth;
      const height = video.videoHeight;
      capture.width = width;
      capture.height = height;
      captureCtx.drawImage(video, 0, 0, width, height);

      try {
        const image = capture.toDataURL("image/jpeg", 0.78);
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image })
        });
        const data = await response.json();
        const stableLabel = smoothLabel(data.label);
        const stableColor = colorMap[stableLabel] || data.color;

        setStatus(stableLabel, stableColor);
        setConfidence(data.confidence);
        faceValue.textContent = data.points && data.points.length ? "Si" : "No";
        tickFps();

        if (data.confidence !== null && data.confidence !== undefined) {
          hint.textContent = `Sistema activo. Confianza del frame: ${Math.round(data.confidence * 100)}%.`;
          hint.classList.remove("error");
        }

        if (data.points && data.points.length) {
          drawLandmarks(data.points, data.width, data.height);
        } else {
          resizeCanvas();
          overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
        }
      } catch (error) {
        hint.textContent = "No se pudo conectar con el servidor de prediccion.";
        hint.classList.add("error");
      } finally {
        processing = false;
      }
    }

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
          audio: false
        });
        video.srcObject = stream;
        labelHistory = [];
        hint.textContent = "Sistema activo.";
        hint.classList.remove("error");
        setStatus("sin rostro", "#ef4444");
        timer = window.setInterval(sendFrame, 180);
      } catch (error) {
        hint.textContent = "El navegador no pudo acceder a la camara.";
        hint.classList.add("error");
      }
    }

    function stopCamera() {
      if (timer) window.clearInterval(timer);
      timer = null;
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
      }
      stream = null;
      labelHistory = [];
      resizeCanvas();
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      setStatus("sin rostro", "#ef4444");
      setConfidence(null);
      faceValue.textContent = "No";
      frameState.textContent = "0 FPS";
      statusPill.textContent = "CAMARA DETENIDA";
      hint.textContent = "Sistema listo para iniciar.";
    }

    startBtn.addEventListener("click", startCamera);
    stopBtn.addEventListener("click", stopCamera);
    window.addEventListener("resize", resizeCanvas);
    resizeCanvas();
  </script>
</body>
</html>
"""


def decode_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)


def landmark_points(landmarks, width, height):
    points = []
    for index in demo_webcam.LANDMARKS_TO_DRAW:
        point = demo_webcam.punto_landmark(landmarks, index, width, height)
        points.append({"x": float(point[0]), "y": float(point[1])})
    return points


def predict_label_and_confidence(features):
    df_features = pd.DataFrame([features])
    df_features = df_features.reindex(columns=columnas_features)

    if hasattr(modelo, "predict_proba"):
        features_scaled = scaler.transform(df_features)
        probabilities = modelo.predict_proba(features_scaled)[0]
        best_index = int(np.argmax(probabilities))
        label = demo_webcam.normalizar_etiqueta(modelo.classes_[best_index])
        return label, float(probabilities[best_index])

    label = demo_webcam.predecir_engagement(modelo, scaler, columnas_features, features)
    return label, None


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/favicon.svg")
def favicon():
    return Response(FAVICON_SVG, mimetype="image/svg+xml")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify(
            {
                "label": "sin rostro",
                "color": COLOR_HEX["sin rostro"],
                "confidence": None,
                "points": [],
            }
        )

    frame = decode_image(image_data)
    if frame is None:
        return jsonify(
            {
                "label": "sin rostro",
                "color": COLOR_HEX["sin rostro"],
                "confidence": None,
                "points": [],
            }
        )

    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with face_mesh_lock:
        results = face_mesh.process(frame_rgb)

    label = "sin rostro"
    confidence = None
    points = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        features = demo_webcam.calcular_features(landmarks, width, height)
        label, confidence = predict_label_and_confidence(features)
        points = landmark_points(landmarks, width, height)

    return jsonify(
        {
            "label": label,
            "color": COLOR_HEX.get(label, COLOR_HEX["sin rostro"]),
            "confidence": confidence,
            "points": points,
            "width": width,
            "height": height,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
