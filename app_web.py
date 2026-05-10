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
from io import BytesIO
import os
import tempfile

from flask import Flask, Response, jsonify, render_template_string, request
import numpy as np
import pandas as pd
from PIL import Image

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / f"engagement_facial_web_mpl_{os.getpid()}"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import mediapipe as mp

import engagement_core


app = Flask(__name__)

modelo, scaler, columnas_features = engagement_core.cargar_recursos_modelo()
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
      --sidebar: #0F2742;
      --primary: #2563EB;
      --primary-soft: #DBEAFE;
      --bg: #F8FAFC;
      --card: #FFFFFF;
      --border: #E2E8F0;
      --text: #0F172A;
      --muted: #64748B;
      --concentrado: #22C55E;
      --confundido: #F97316;
      --aburrido: #9CA3AF;
      --sorprendido: #EAB308;
      --sin-rostro: #EF4444;
      --state-color: #EF4444;
      --state-soft: rgba(239, 68, 68, 0.12);
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, Segoe UI, Arial, Helvetica, sans-serif;
    }

    .app {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      position: sticky;
      top: 0;
      height: 100vh;
      background: linear-gradient(180deg, #0F2742 0%, #0B1B2F 100%);
      color: #FFFFFF;
      padding: 26px 20px;
      display: flex;
      flex-direction: column;
      gap: 24px;
      box-shadow: 18px 0 45px rgba(15, 39, 66, 0.18);
      z-index: 2;
    }

    .brand {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .brand-mark {
      width: 56px;
      height: 56px;
      border-radius: 18px;
      display: grid;
      place-items: center;
      background: rgba(37, 99, 235, 0.16);
      border: 1px solid rgba(219, 234, 254, 0.22);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.05);
      font-size: 28px;
    }

    .brand h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.1;
      letter-spacing: 0;
    }

    .brand p {
      margin: 6px 0 0;
      color: #BFDBFE;
      font-size: 12px;
      line-height: 1.35;
    }

    .menu {
      display: grid;
      gap: 8px;
      padding-top: 8px;
      border-top: 1px solid rgba(226, 232, 240, 0.14);
    }

    .menu a {
      display: flex;
      align-items: center;
      gap: 11px;
      min-height: 44px;
      padding: 0 13px;
      border-radius: 12px;
      color: #CBD5E1;
      text-decoration: none;
      font-size: 14px;
      font-weight: 700;
      transition: background 180ms ease, color 180ms ease, transform 180ms ease;
    }

    .menu a:hover,
    .menu a.active {
      background: var(--primary);
      color: #FFFFFF;
      transform: translateX(2px);
      box-shadow: 0 12px 28px rgba(37, 99, 235, 0.28);
    }

    .menu-icon {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.08);
      flex: 0 0 auto;
    }

    .academic-card {
      margin-top: auto;
      border: 1px solid rgba(219, 234, 254, 0.18);
      border-radius: 18px;
      background: rgba(37, 99, 235, 0.14);
      padding: 18px;
    }

    .academic-card strong,
    .academic-card span {
      display: block;
    }

    .academic-card strong {
      font-size: 15px;
      margin: 8px 0;
    }

    .academic-card span {
      color: #BFDBFE;
      font-size: 13px;
      line-height: 1.45;
    }

    .main {
      min-width: 0;
      padding: 28px;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: center;
      margin-bottom: 22px;
    }

    .welcome {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .welcome-icon {
      width: 58px;
      height: 58px;
      border-radius: 19px;
      display: grid;
      place-items: center;
      color: #FFFFFF;
      background: linear-gradient(135deg, #2563EB, #3B82F6);
      box-shadow: 0 18px 35px rgba(37, 99, 235, 0.28);
      font-size: 28px;
    }

    .welcome h2 {
      margin: 0;
      color: var(--primary);
      font-size: 30px;
      line-height: 1.05;
      letter-spacing: 0;
    }

    .welcome p {
      max-width: 680px;
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }

    .badges {
      display: grid;
      grid-template-columns: repeat(4, max-content);
      gap: 10px;
    }

    .badge {
      min-width: 118px;
      min-height: 58px;
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: var(--card);
      padding: 10px 13px;
      box-shadow: var(--shadow);
    }

    .badge-icon {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: #ECFDF5;
      color: var(--concentrado);
      font-size: 17px;
      flex: 0 0 auto;
    }

    .badge strong,
    .badge span {
      display: block;
      white-space: nowrap;
    }

    .badge strong {
      color: var(--primary);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .badge span {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }

    .kpis {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }

    .kpi-card,
    .panel,
    .info-card {
      border: 1px solid var(--border);
      border-radius: 18px;
      background: var(--card);
      box-shadow: var(--shadow);
    }

    .kpi-card {
      min-height: 112px;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 13px;
      align-items: center;
      padding: 18px;
      transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .kpi-card:hover,
    .panel:hover,
    .info-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 22px 55px rgba(15, 23, 42, 0.11);
    }

    .kpi-icon {
      width: 54px;
      height: 54px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 26px;
    }

    .kpi-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .kpi-card strong {
      display: block;
      margin-top: 7px;
      color: var(--text);
      font-size: 21px;
      font-weight: 900;
    }

    .kpi-card small {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }

    .panel {
      overflow: hidden;
      transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .panel-title {
      display: flex;
      align-items: center;
      gap: 9px;
      margin: 0;
      padding: 18px 20px 0;
      color: var(--primary);
      font-size: 14px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .video-frame {
      position: relative;
      margin: 16px 18px 18px;
      border: 4px solid var(--state-color);
      border-radius: 18px;
      overflow: hidden;
      background: #0F172A;
      aspect-ratio: 16 / 9;
      transition: border-color 180ms ease, box-shadow 180ms ease;
      box-shadow: 0 0 0 7px var(--state-soft);
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

    .video-badge,
    .fps-chip {
      position: absolute;
      z-index: 3;
      min-height: 32px;
      display: inline-flex;
      align-items: center;
      border-radius: 10px;
      padding: 0 12px;
      font-size: 12px;
      font-weight: 900;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.18);
    }

    .video-badge {
      top: 14px;
      left: 14px;
      gap: 8px;
      color: var(--state-color);
      background: rgba(255, 255, 255, 0.9);
    }

    .video-badge::before {
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--state-color);
    }

    .fps-chip {
      top: 14px;
      right: 14px;
      color: #FFFFFF;
      background: rgba(15, 23, 42, 0.76);
    }

    .engagement-overlay {
      position: absolute;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      width: min(86%, 590px);
      z-index: 3;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 16px;
      border: 1px solid rgba(226, 232, 240, 0.85);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(12px);
      padding: 13px 18px;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
    }

    .face-emote {
      width: 58px;
      height: 58px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--state-color);
      color: #FFFFFF;
      font-size: 28px;
      flex: 0 0 auto;
    }

    .overlay-copy span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .overlay-copy strong {
      display: block;
      color: var(--state-color);
      font-size: 28px;
      line-height: 1.1;
      font-weight: 950;
    }

    .overlay-confidence {
      min-width: 96px;
      border-left: 1px solid var(--border);
      padding-left: 16px;
    }

    .overlay-confidence span,
    .overlay-confidence strong {
      display: block;
      text-align: center;
    }

    .overlay-confidence span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .overlay-confidence strong {
      margin-top: 4px;
      color: var(--state-color);
      font-size: 22px;
      font-weight: 950;
    }

    .right-stack {
      display: grid;
      gap: 16px;
    }

    .panel-section {
      padding: 18px;
      border-bottom: 1px solid var(--border);
    }

    .panel-section:last-child { border-bottom: 0; }

    .section-title {
      margin: 0 0 14px;
      color: var(--primary);
      font-size: 13px;
      font-weight: 950;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .state-card {
      border: 1px solid color-mix(in srgb, var(--state-color), white 62%);
      border-radius: 16px;
      background: var(--state-soft);
      padding: 18px;
    }

    .state-heading {
      display: flex;
      align-items: center;
      gap: 14px;
    }

    .state-icon {
      width: 54px;
      height: 54px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--state-color);
      color: #FFFFFF;
      font-size: 26px;
    }

    .state-label {
      color: var(--state-color);
      font-size: 27px;
      font-weight: 950;
      line-height: 1;
    }

    .state-description {
      margin: 12px 0 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      font-weight: 650;
    }

    .state-meta {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.78);
      padding: 12px;
      min-height: 74px;
    }

    .metric span,
    .metric strong {
      display: block;
    }

    .metric span {
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
    }

    .metric strong {
      margin-top: 7px;
      color: var(--text);
      font-size: 20px;
      font-weight: 950;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    button {
      min-height: 46px;
      border: 0;
      border-radius: 12px;
      color: #FFFFFF;
      font-size: 14px;
      font-weight: 900;
      cursor: pointer;
      transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
    }

    button:hover {
      transform: translateY(-1px);
      filter: brightness(1.02);
    }

    .primary {
      background: var(--primary);
      box-shadow: 0 12px 28px rgba(37, 99, 235, 0.25);
    }

    .danger {
      background: #FFFFFF;
      color: var(--sin-rostro);
      border: 1px solid #FCA5A5;
    }

    .hint {
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .error { color: var(--sin-rostro); }

    .legend {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .legend-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 9px;
      align-items: center;
    }

    .legend-dot {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      color: #FFFFFF;
      background: var(--dot-color);
      font-size: 14px;
    }

    .legend-row strong,
    .legend-row span {
      display: block;
    }

    .legend-row strong {
      color: var(--text);
      font-size: 13px;
      font-weight: 900;
    }

    .legend-row span {
      margin-top: 2px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
    }

    .system-list {
      display: grid;
      gap: 10px;
      margin: 0;
      padding: 0;
      list-style: none;
    }

    .system-list li {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }

    .system-list strong {
      color: var(--text);
      text-align: right;
      font-weight: 900;
    }

    .flow-panel {
      margin-top: 18px;
      padding: 20px;
    }

    .flow {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      align-items: stretch;
    }

    .flow-step {
      position: relative;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #FFFFFF;
      padding: 16px;
      min-height: 126px;
    }

    .flow-step:not(:last-child)::after {
      content: ">";
      position: absolute;
      right: -12px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--primary);
      font-weight: 950;
      font-size: 18px;
      z-index: 1;
    }

    .flow-icon {
      width: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      color: #FFFFFF;
      background: var(--primary);
      font-size: 20px;
      margin-bottom: 12px;
    }

    .flow-step strong,
    .flow-step span {
      display: block;
    }

    .flow-step strong {
      color: var(--text);
      font-size: 14px;
      font-weight: 950;
    }

    .flow-step span {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      font-weight: 700;
    }

    .education-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 18px;
    }

    .info-card {
      padding: 18px;
      transition: transform 180ms ease, box-shadow 180ms ease;
    }

    .info-card .info-icon {
      width: 42px;
      height: 42px;
      display: grid;
      place-items: center;
      border-radius: 13px;
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 22px;
      margin-bottom: 12px;
    }

    .info-card h3 {
      margin: 0;
      color: var(--text);
      font-size: 16px;
      font-weight: 950;
    }

    .info-card p {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      font-weight: 650;
    }

    @media (max-width: 1180px) {
      .app { grid-template-columns: 240px minmax(0, 1fr); }
      .hero { grid-template-columns: 1fr; }
      .badges { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .dashboard-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      .sidebar {
        position: relative;
        height: auto;
      }
      .menu { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .academic-card { margin-top: 0; }
      .main { padding: 18px; }
      .flow { grid-template-columns: 1fr; }
      .flow-step:not(:last-child)::after { display: none; }
      .education-grid { grid-template-columns: 1fr; }
      .engagement-overlay {
        width: calc(100% - 24px);
        grid-template-columns: auto minmax(0, 1fr);
      }
      .overlay-confidence {
        grid-column: 1 / -1;
        border-left: 0;
        border-top: 1px solid var(--border);
        padding: 10px 0 0;
      }
    }

    @media (max-width: 620px) {
      .badges,
      .kpis,
      .legend,
      .state-meta,
      .controls { grid-template-columns: 1fr; }
      .welcome { align-items: flex-start; }
      .welcome h2 { font-size: 24px; }
      .menu { grid-template-columns: 1fr; }
      .overlay-copy strong { font-size: 22px; }
      .face-emote { width: 48px; height: 48px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">🎓</div>
        <div>
          <h1>EngageFace Aula</h1>
          <p>Medidor de engagement facial en aulas virtuales</p>
        </div>
      </div>

      <nav class="menu" aria-label="Menu principal">
        <a class="active" href="#dashboard"><span class="menu-icon">🏠</span>Dashboard</a>
        <a href="#monitor"><span class="menu-icon">👁</span>En tiempo real</a>
        <a href="#historial"><span class="menu-icon">🕘</span>Historial</a>
        <a href="#analisis"><span class="menu-icon">📊</span>Análisis</a>
        <a href="#reportes"><span class="menu-icon">📄</span>Reportes</a>
        <a href="#configuracion"><span class="menu-icon">⚙</span>Configuración</a>
        <a href="#informacion"><span class="menu-icon">ⓘ</span>Información</a>
      </nav>

      <div class="academic-card">
        <div aria-hidden="true">👥</div>
        <strong>Proyecto Académico</strong>
        <span>IA aplicada a la educación</span>
        <span>Universidad 2025</span>
      </div>
    </aside>

    <main id="dashboard" class="main">
      <header class="hero">
        <div class="welcome">
          <div class="welcome-icon" aria-hidden="true">🎓</div>
          <div>
            <h2>¡Bienvenido!</h2>
            <p>Sistema inteligente que analiza expresiones faciales para apoyar el aprendizaje y bienestar del estudiante.</p>
          </div>
        </div>

        <div class="badges" aria-label="Indicadores rapidos">
          <div class="badge"><div class="badge-icon">🧠</div><div><strong>IA activa</strong><span>Inferencia en tiempo real</span></div></div>
          <div class="badge"><div class="badge-icon">📶</div><div><strong>Tiempo real</strong><span>Conectado</span></div></div>
          <div class="badge"><div class="badge-icon">〽</div><div><strong>FPS</strong><span id="frameState">0 FPS</span></div></div>
          <div class="badge"><div class="badge-icon">●</div><div><strong>Sistema</strong><span id="systemValue">Listo</span></div></div>
        </div>
      </header>

      <section class="kpis" aria-label="Indicadores principales">
        <article class="kpi-card"><div class="kpi-icon">🧠</div><div><span>Modelo actual</span><strong>Gradient Boosting</strong><small>Clasificación multiclase</small></div></article>
        <article class="kpi-card"><div class="kpi-icon">⌘</div><div><span>Features utilizadas</span><strong>11</strong><small>Razones geométricas</small></div></article>
        <article class="kpi-card"><div class="kpi-icon">🗄</div><div><span>Dataset procesado</span><strong>9,217</strong><small>Muestras analizadas</small></div></article>
        <article class="kpi-card"><div class="kpi-icon">🛡</div><div><span>Estado del sistema</span><strong id="kpiSystemValue">Activo</strong><small>Todo funcionando correctamente</small></div></article>
      </section>

      <div class="dashboard-grid">
        <section id="monitor" class="panel" aria-label="Monitoreo en tiempo real">
          <h2 class="panel-title">📹 Monitoreo en tiempo real</h2>
          <div id="videoFrame" class="video-frame">
            <video id="video" autoplay playsinline muted></video>
            <canvas id="overlay"></canvas>
            <div id="faceBadge" class="video-badge">SIN ROSTRO</div>
            <div class="fps-chip" id="fpsChip">0 FPS</div>
            <div class="engagement-overlay">
              <div id="faceEmote" class="face-emote">🙂</div>
              <div class="overlay-copy">
                <span>Engagement</span>
                <strong id="overlayLabel">SIN ROSTRO</strong>
              </div>
              <div class="overlay-confidence">
                <span>Confianza</span>
                <strong id="overlayConfidence">--</strong>
              </div>
            </div>
          </div>
        </section>

        <aside class="right-stack" aria-label="Panel derecho">
          <section class="panel">
            <div class="panel-section">
              <h2 class="section-title">Estado actual</h2>
              <div class="state-card">
                <div class="state-heading">
                  <div id="stateIcon" class="state-icon">🙂</div>
                  <div id="stateLabel" class="state-label">SIN ROSTRO</div>
                </div>
                <p id="stateDescription" class="state-description">No se detecta un rostro frente a la cámara.</p>
                <div class="state-meta">
                  <div class="metric"><span>Confianza</span><strong id="confidenceValue">--</strong></div>
                  <div class="metric"><span>Rostro</span><strong id="faceValue">No</strong></div>
                </div>
              </div>
            </div>

            <div class="panel-section">
              <h2 class="section-title">Controles</h2>
              <div class="controls">
                <button id="startBtn" class="primary" type="button">📹 Iniciar cámara</button>
                <button id="stopBtn" class="danger" type="button">■ Detener</button>
              </div>
              <p id="hint" class="hint">Sistema listo para iniciar.</p>
            </div>
          </section>

          <section class="panel">
            <div class="panel-section">
              <h2 class="section-title">Leyenda de estados</h2>
              <div class="legend">
                <div class="legend-row"><span class="legend-dot" style="--dot-color:#22C55E">🙂</span><div><strong>Concentrado</strong><span>Atento y enfocado</span></div></div>
                <div class="legend-row"><span class="legend-dot" style="--dot-color:#F97316">🤔</span><div><strong>Confundido</strong><span>Duda o dificultad</span></div></div>
                <div class="legend-row"><span class="legend-dot" style="--dot-color:#9CA3AF">😐</span><div><strong>Aburrido</strong><span>Poca estimulación</span></div></div>
                <div class="legend-row"><span class="legend-dot" style="--dot-color:#EAB308">😮</span><div><strong>Sorprendido</strong><span>Alta sorpresa</span></div></div>
              </div>
            </div>

            <div class="panel-section">
              <h2 class="section-title">Sistema</h2>
              <ul class="system-list">
                <li><span>Backend</span><strong>Flask + Gunicorn</strong></li>
                <li><span>Visión por computadora</span><strong>MediaPipe FaceMesh</strong></li>
                <li><span>Deployment</span><strong>Railway</strong></li>
                <li><span>Versión</span><strong>1.0.0</strong></li>
              </ul>
            </div>
          </section>
        </aside>
      </div>

      <section id="analisis" class="panel flow-panel">
        <h2 class="section-title">¿Cómo funciona el sistema?</h2>
        <div class="flow">
          <div class="flow-step"><div class="flow-icon">📹</div><strong>1. Captura</strong><span>Webcam en tiempo real</span></div>
          <div class="flow-step"><div class="flow-icon">🧩</div><strong>2. Detección</strong><span>FaceMesh 468 puntos</span></div>
          <div class="flow-step"><div class="flow-icon">⚙</div><strong>3. Extracción</strong><span>11 razones geométricas</span></div>
          <div class="flow-step"><div class="flow-icon">🧠</div><strong>4. Predicción</strong><span>Modelo de ML entrenado</span></div>
          <div class="flow-step"><div class="flow-icon">📈</div><strong>5. Resultado</strong><span>Estado de engagement</span></div>
        </div>
      </section>

      <section id="informacion" class="education-grid">
        <article class="info-card"><div class="info-icon">📘</div><h3>Uso educativo</h3><p>Esta herramienta ayuda a docentes a identificar patrones de engagement y mejorar estrategias pedagógicas.</p></article>
        <article class="info-card"><div class="info-icon">🛡</div><h3>Ética y privacidad</h3><p>Los datos se procesan en tiempo real y no se almacenan imágenes ni información personal.</p></article>
        <article class="info-card"><div class="info-icon">ⓘ</div><h3>Nota importante</h3><p>Este sistema es un apoyo al docente, no reemplaza su criterio profesional.</p></article>
      </section>

      <canvas id="capture" style="display:none;"></canvas>
    </main>
  </div>

  <script>
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const capture = document.getElementById("capture");
    const overlayLabel = document.getElementById("overlayLabel");
    const overlayConfidence = document.getElementById("overlayConfidence");
    const faceBadge = document.getElementById("faceBadge");
    const fpsChip = document.getElementById("fpsChip");
    const videoFrame = document.getElementById("videoFrame");
    const faceEmote = document.getElementById("faceEmote");
    const stateIcon = document.getElementById("stateIcon");
    const stateLabel = document.getElementById("stateLabel");
    const stateDescription = document.getElementById("stateDescription");
    const confidenceValue = document.getElementById("confidenceValue");
    const faceValue = document.getElementById("faceValue");
    const frameState = document.getElementById("frameState");
    const systemValue = document.getElementById("systemValue");
    const kpiSystemValue = document.getElementById("kpiSystemValue");
    const hint = document.getElementById("hint");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");

    const overlayCtx = overlay.getContext("2d");
    const captureCtx = capture.getContext("2d");
    const colorMap = {
      "concentrado": "#22C55E",
      "confundido": "#F97316",
      "aburrido": "#9CA3AF",
      "sorprendido": "#EAB308",
      "sin rostro": "#EF4444"
    };
    const stateConfig = {
      "concentrado": {
        icon: "🙂",
        description: "El estudiante muestra señales de atención y enfoque en la actividad."
      },
      "confundido": {
        icon: "🤔",
        description: "El estudiante podría presentar duda o dificultad con el contenido."
      },
      "aburrido": {
        icon: "😐",
        description: "El estudiante muestra baja estimulación o poco interés visual."
      },
      "sorprendido": {
        icon: "😮",
        description: "El estudiante presenta una reacción de alta sorpresa."
      },
      "sin rostro": {
        icon: "🙂",
        description: "No se detecta un rostro frente a la cámara."
      }
    };

    let stream = null;
    let timer = null;
    let processing = false;
    let labelHistory = [];
    let frames = 0;
    let lastFpsTime = performance.now();

    function setStatus(label, color) {
      const upper = label.toUpperCase();
      const config = stateConfig[label] || stateConfig["sin rostro"];
      overlayLabel.textContent = upper;
      stateLabel.textContent = upper;
      stateDescription.textContent = config.description;
      faceEmote.textContent = config.icon;
      stateIcon.textContent = config.icon;
      faceBadge.textContent = label === "sin rostro" ? "SIN ROSTRO" : "ROSTRO DETECTADO";
      stateLabel.style.color = color;
      document.documentElement.style.setProperty("--state-color", color);
      document.documentElement.style.setProperty("--state-soft", `${color}1F`);
      videoFrame.style.borderColor = color;
    }

    function setConfidence(value) {
      const text = value === null || value === undefined ? "--" : `${Math.round(value * 100)}%`;
      confidenceValue.textContent = text;
      overlayConfidence.textContent = text;
    }

    function tickFps() {
      frames += 1;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        const fpsText = `${frames} FPS`;
        frameState.textContent = fpsText;
        fpsChip.textContent = fpsText;
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
        systemValue.textContent = data.points && data.points.length ? "Activo" : "Buscando";
        kpiSystemValue.textContent = data.points && data.points.length ? "Activo" : "Buscando";
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
        systemValue.textContent = "Permiso";
        kpiSystemValue.textContent = "Permiso";
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
      fpsChip.textContent = "0 FPS";
      systemValue.textContent = "Listo";
      kpiSystemValue.textContent = "Activo";
      faceBadge.textContent = "SIN ROSTRO";
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
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def landmark_points(landmarks, width, height):
    points = []
    for index in engagement_core.LANDMARKS_TO_DRAW:
        point = engagement_core.punto_landmark(landmarks, index, width, height)
        points.append({"x": float(point[0]), "y": float(point[1])})
    return points


def predict_label_and_confidence(features):
    df_features = pd.DataFrame([features])
    df_features = df_features.reindex(columns=columnas_features)

    if hasattr(modelo, "predict_proba"):
        features_scaled = scaler.transform(df_features)
        probabilities = modelo.predict_proba(features_scaled)[0]
        best_index = int(np.argmax(probabilities))
        label = engagement_core.normalizar_etiqueta(modelo.classes_[best_index])
        return label, float(probabilities[best_index])

    label = engagement_core.predecir_engagement(modelo, scaler, columnas_features, features)
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
    frame_rgb = frame

    with face_mesh_lock:
        results = face_mesh.process(frame_rgb)

    label = "sin rostro"
    confidence = None
    points = []
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        features = engagement_core.calcular_features(landmarks, width, height)
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
    app.run(host="0.0.0.0", port=port, debug=False)
