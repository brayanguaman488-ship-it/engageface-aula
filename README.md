# EngageFace Aula

Aplicacion web para medir engagement facial en aulas virtuales usando OpenCV, MediaPipe FaceMesh y un modelo supervisado entrenado con features geometricas.

## Ejecutar localmente

```powershell
cd C:\Proyecto_Engagement_Facial
python -m pip install -r requirements.txt
python app_web.py
```

Abrir:

```text
http://127.0.0.1:5000
```

## Archivos principales

- `app_web.py`: aplicacion web Flask.
- `demo_webcam.py`: version local con ventana OpenCV.
- `entrenar_modelo_engagement.py`: entrenamiento y comparacion de modelos.
- `modelo_engagement.pkl`: modelo ganador exportado.
- `scaler_engagement.pkl`: escalador usado por el modelo.
- `columnas_features.pkl`: orden de features.
- `dataset_engagement.csv`: dataset procesado.

## Railway

Railway puede iniciar la app con el comando definido en `Procfile`:

```text
web: python app_web.py
```

La aplicacion usa la variable `PORT` cuando Railway la define.
