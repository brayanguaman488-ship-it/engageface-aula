"""
demo_webcam.py
Proyecto 2 - Medidor de Engagement Facial

Demostracion funcional en tiempo real:
- Abre la webcam con OpenCV.
- Detecta un rostro usando MediaPipe FaceMesh.
- Extrae las mismas features geometricas usadas durante el entrenamiento.
- Carga el modelo, el scaler y el orden de columnas guardados.
- Muestra en pantalla el estado estimado de engagement del estudiante.

Estados posibles:
- concentrado
- confundido
- aburrido
- sorprendido

Uso:
    python demo_webcam.py

Presione la tecla 'q' para cerrar la ventana.
"""

from pathlib import Path
import os
import sys
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / f"engagement_facial_mpl_{os.getpid()}"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import mediapipe as mp

MODEL_FILE = BASE_DIR / "modelo_engagement.pkl"
SCALER_FILE = BASE_DIR / "scaler_engagement.pkl"
FEATURES_FILE = BASE_DIR / "columnas_features.pkl"

FEATURE_NAMES = [
    "apertura_ojo_izq",
    "apertura_ojo_der",
    "apertura_ocular_promedio",
    "curvatura_boca",
    "posicion_ceja_izq",
    "posicion_ceja_der",
    "posicion_cejas_promedio",
    "simetria_ocular",
    "simetria_cejas",
    "tension_boca",
    "ancho_boca_normalizado",
]

LABEL_COLORS = {
    "concentrado": (0, 200, 0),
    "confundido": (0, 140, 255),
    "aburrido": (135, 135, 135),
    "sorprendido": (0, 255, 255),
    "sin rostro": (0, 0, 255),
}

LANDMARK_GROUPS = {
    "ojo_izq": [33, 133, 145, 159],
    "ojo_der": [362, 263, 374, 386],
    "boca": [78, 308, 13, 14],
    "cejas": [105, 334],
    "referencia_rostro": [1, 152],
}

LANDMARKS_TO_DRAW = [
    33,
    133,
    145,
    159,
    362,
    263,
    374,
    386,
    78,
    308,
    13,
    14,
    105,
    334,
    1,
    152,
]


def cargar_archivo_pickle(path):
    """Carga un archivo pickle/joblib y entrega un error claro si falta."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo requerido: {path.name}")
    return joblib.load(path)


def distancia(punto_a, punto_b):
    return float(np.linalg.norm(punto_a - punto_b))


def punto_landmark(landmarks, indice, ancho_frame, alto_frame):
    landmark = landmarks[indice]
    return np.array(
        [landmark.x * ancho_frame, landmark.y * alto_frame],
        dtype=np.float32,
    )


def distancia_segura(punto_a, punto_b, minimo=1e-6):
    return max(distancia(punto_a, punto_b), minimo)


def calcular_features(landmarks, ancho_frame, alto_frame):
    """Calcula las features geometricas usadas por el modelo entrenado."""
    ojo_izq_ext = punto_landmark(landmarks, 33, ancho_frame, alto_frame)
    ojo_izq_int = punto_landmark(landmarks, 133, ancho_frame, alto_frame)
    ojo_izq_inf = punto_landmark(landmarks, 145, ancho_frame, alto_frame)
    ojo_izq_sup = punto_landmark(landmarks, 159, ancho_frame, alto_frame)

    ojo_der_int = punto_landmark(landmarks, 362, ancho_frame, alto_frame)
    ojo_der_ext = punto_landmark(landmarks, 263, ancho_frame, alto_frame)
    ojo_der_inf = punto_landmark(landmarks, 374, ancho_frame, alto_frame)
    ojo_der_sup = punto_landmark(landmarks, 386, ancho_frame, alto_frame)

    boca_izq = punto_landmark(landmarks, 78, ancho_frame, alto_frame)
    boca_der = punto_landmark(landmarks, 308, ancho_frame, alto_frame)
    boca_sup = punto_landmark(landmarks, 13, ancho_frame, alto_frame)
    boca_inf = punto_landmark(landmarks, 14, ancho_frame, alto_frame)

    ceja_izq = punto_landmark(landmarks, 105, ancho_frame, alto_frame)
    ceja_der = punto_landmark(landmarks, 334, ancho_frame, alto_frame)

    referencia_sup = punto_landmark(landmarks, 1, ancho_frame, alto_frame)
    referencia_inf = punto_landmark(landmarks, 152, ancho_frame, alto_frame)
    altura_rostro = distancia_segura(referencia_sup, referencia_inf)

    ancho_ojo_izq = distancia_segura(ojo_izq_ext, ojo_izq_int)
    ancho_ojo_der = distancia_segura(ojo_der_int, ojo_der_ext)
    apertura_ojo_izq = distancia(ojo_izq_sup, ojo_izq_inf) / ancho_ojo_izq
    apertura_ojo_der = distancia(ojo_der_sup, ojo_der_inf) / ancho_ojo_der
    apertura_ocular_promedio = (apertura_ojo_izq + apertura_ojo_der) / 2.0

    ancho_boca = distancia_segura(boca_izq, boca_der)
    apertura_boca = distancia(boca_sup, boca_inf)
    curvatura_boca = apertura_boca / ancho_boca

    posicion_ceja_izq = distancia(ceja_izq, ojo_izq_sup) / altura_rostro
    posicion_ceja_der = distancia(ceja_der, ojo_der_sup) / altura_rostro
    posicion_cejas_promedio = (posicion_ceja_izq + posicion_ceja_der) / 2.0

    simetria_ocular = abs(apertura_ojo_izq - apertura_ojo_der)
    simetria_cejas = abs(posicion_ceja_izq - posicion_ceja_der)
    tension_boca = apertura_boca / altura_rostro
    ancho_boca_normalizado = ancho_boca / altura_rostro

    return {
        "apertura_ojo_izq": apertura_ojo_izq,
        "apertura_ojo_der": apertura_ojo_der,
        "apertura_ocular_promedio": apertura_ocular_promedio,
        "curvatura_boca": curvatura_boca,
        "posicion_ceja_izq": posicion_ceja_izq,
        "posicion_ceja_der": posicion_ceja_der,
        "posicion_cejas_promedio": posicion_cejas_promedio,
        "simetria_ocular": simetria_ocular,
        "simetria_cejas": simetria_cejas,
        "tension_boca": tension_boca,
        "ancho_boca_normalizado": ancho_boca_normalizado,
    }


def normalizar_etiqueta(prediccion):
    if isinstance(prediccion, (list, tuple, np.ndarray)):
        prediccion = prediccion[0]

    if isinstance(prediccion, bytes):
        prediccion = prediccion.decode("utf-8", errors="ignore")

    if isinstance(prediccion, str):
        return prediccion.strip().lower()

    if isinstance(prediccion, (int, np.integer)):
        etiquetas_por_indice = {
            0: "concentrado",
            1: "confundido",
            2: "aburrido",
            3: "sorprendido",
        }
        return etiquetas_por_indice.get(int(prediccion), "sin rostro")

    return str(prediccion).strip().lower()


def predecir_engagement(modelo, scaler, columnas_features, features):
    df_features = pd.DataFrame([features])
    df_features = df_features.reindex(columns=columnas_features)

    if df_features.isnull().any().any():
        columnas_faltantes = df_features.columns[df_features.isnull().any()].tolist()
        raise ValueError(f"Faltan features requeridas: {columnas_faltantes}")

    features_escaladas = scaler.transform(df_features)
    prediccion = modelo.predict(features_escaladas)
    etiqueta = normalizar_etiqueta(prediccion)
    return etiqueta if etiqueta in LABEL_COLORS else "sin rostro"


def dibujar_caja_estado(frame, etiqueta):
    color = LABEL_COLORS.get(etiqueta, LABEL_COLORS["sin rostro"])
    alto_caja = 68

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], alto_caja), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    texto = f"Engagement: {etiqueta.upper()}"
    cv2.putText(
        frame,
        texto,
        (22, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        color,
        2,
        cv2.LINE_AA,
    )


def dibujar_landmarks_ligeros(frame, landmarks, ancho_frame, alto_frame):
    color_punto = (235, 235, 235)
    color_linea = (80, 220, 220)

    for indice in LANDMARKS_TO_DRAW:
        punto = punto_landmark(landmarks, indice, ancho_frame, alto_frame)
        cv2.circle(frame, (int(punto[0]), int(punto[1])), 2, color_punto, -1, cv2.LINE_AA)

    conexiones = [
        (33, 133),
        (145, 159),
        (362, 263),
        (374, 386),
        (78, 308),
        (13, 14),
        (1, 152),
    ]
    for inicio, fin in conexiones:
        p1 = punto_landmark(landmarks, inicio, ancho_frame, alto_frame)
        p2 = punto_landmark(landmarks, fin, ancho_frame, alto_frame)
        cv2.line(
            frame,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color_linea,
            1,
            cv2.LINE_AA,
        )


def validar_columnas(columnas_features):
    columnas_features = list(columnas_features)
    faltantes = [col for col in FEATURE_NAMES if col not in columnas_features]
    if faltantes:
        raise ValueError(f"columnas_features.pkl no contiene estas columnas: {faltantes}")
    return columnas_features


def cargar_recursos_modelo():
    try:
        modelo = cargar_archivo_pickle(MODEL_FILE)
        scaler = cargar_archivo_pickle(SCALER_FILE)
        columnas_features = validar_columnas(cargar_archivo_pickle(FEATURES_FILE))
        return modelo, scaler, columnas_features
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
    except Exception as exc:
        print(f"ERROR al cargar los recursos del modelo: {exc}")

    sys.exit(1)


def abrir_camara():
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("ERROR: No se pudo abrir la camara. Verifique conexion y permisos.")
        sys.exit(1)
    return camara


def main():
    modelo, scaler, columnas_features = cargar_recursos_modelo()
    camara = abrir_camara()

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("ERROR: No se pudo leer imagen desde la camara.")
                break

            frame = cv2.flip(frame, 1)
            alto_frame, ancho_frame = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultados = face_mesh.process(frame_rgb)

            etiqueta = "sin rostro"
            if resultados.multi_face_landmarks:
                landmarks = resultados.multi_face_landmarks[0].landmark
                try:
                    features = calcular_features(landmarks, ancho_frame, alto_frame)
                    etiqueta = predecir_engagement(modelo, scaler, columnas_features, features)
                except Exception as exc:
                    print(f"WARNING: No se pudo predecir en este frame: {exc}")
                    etiqueta = "sin rostro"

                dibujar_landmarks_ligeros(frame, landmarks, ancho_frame, alto_frame)

            dibujar_caja_estado(frame, etiqueta)
            cv2.imshow("Proyecto 2 - Medidor de Engagement Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
