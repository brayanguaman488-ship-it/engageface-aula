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
web: gunicorn app_web:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
```

La aplicacion usa la variable `PORT` cuando Railway la define.

### Base de datos PostgreSQL en Railway

1. En Railway, abre el proyecto de la app.
2. Selecciona **New** -> **Database** -> **PostgreSQL**.
3. Railway creara la variable `DATABASE_URL` automaticamente para la app si ambos servicios estan en el mismo proyecto.
4. Despliega de nuevo la app.

Cuando `DATABASE_URL` existe, la app crea automaticamente la tabla:

```sql
engagement_predictions (
  id,
  created_at,
  label,
  confidence,
  face_detected,
  frame_width,
  frame_height,
  points_count,
  client_ip,
  user_agent
)
```

Para evitar llenar la base con demasiados frames, por defecto se guarda 1 de cada 10 predicciones. Puedes cambiarlo en Railway con:

```text
SAVE_EVERY_N_PREDICTIONS=5
```

El estado de conexion se puede revisar en:

```text
/db-status
```
