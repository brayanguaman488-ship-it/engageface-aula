from pathlib import Path
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / f"engagement_facial_mpl_{os.getpid()}"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

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

    apertura_ojo_izq = distancia(ojo_izq_sup, ojo_izq_inf) / distancia_segura(ojo_izq_ext, ojo_izq_int)
    apertura_ojo_der = distancia(ojo_der_sup, ojo_der_inf) / distancia_segura(ojo_der_int, ojo_der_ext)
    apertura_ocular_promedio = (apertura_ojo_izq + apertura_ojo_der) / 2.0

    ancho_boca = distancia_segura(boca_izq, boca_der)
    apertura_boca = distancia(boca_sup, boca_inf)
    curvatura_boca = apertura_boca / ancho_boca

    posicion_ceja_izq = distancia(ceja_izq, ojo_izq_sup) / altura_rostro
    posicion_ceja_der = distancia(ceja_der, ojo_der_sup) / altura_rostro
    posicion_cejas_promedio = (posicion_ceja_izq + posicion_ceja_der) / 2.0

    return {
        "apertura_ojo_izq": apertura_ojo_izq,
        "apertura_ojo_der": apertura_ojo_der,
        "apertura_ocular_promedio": apertura_ocular_promedio,
        "curvatura_boca": curvatura_boca,
        "posicion_ceja_izq": posicion_ceja_izq,
        "posicion_ceja_der": posicion_ceja_der,
        "posicion_cejas_promedio": posicion_cejas_promedio,
        "simetria_ocular": abs(apertura_ojo_izq - apertura_ojo_der),
        "simetria_cejas": abs(posicion_ceja_izq - posicion_ceja_der),
        "tension_boca": apertura_boca / altura_rostro,
        "ancho_boca_normalizado": ancho_boca / altura_rostro,
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
