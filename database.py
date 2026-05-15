"""
database.py
Conexion opcional a PostgreSQL para guardar predicciones en Railway.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
import threading

try:
    import psycopg2
    from psycopg2 import pool
except ImportError:  # Permite ejecutar la app sin la dependencia instalada.
    psycopg2 = None
    pool = None


DATABASE_URL = os.environ.get("DATABASE_URL")
SAVE_EVERY_N_PREDICTIONS = max(
    1, int(os.environ.get("SAVE_EVERY_N_PREDICTIONS", "10"))
)

_connection_pool = None
_pool_lock = threading.Lock()
_prediction_counter = 0
_counter_lock = threading.Lock()
_last_error = None


def is_enabled():
    return bool(DATABASE_URL and psycopg2)


def last_error():
    return _last_error


def _set_last_error(error):
    global _last_error
    _last_error = str(error) if error else None


def init_db():
    if not is_enabled():
        return False

    try:
        global _connection_pool
        with _pool_lock:
            if _connection_pool is None:
                _connection_pool = pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=int(os.environ.get("DB_POOL_MAX_CONN", "5")),
                    dsn=DATABASE_URL,
                )

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS engagement_predictions (
                        id BIGSERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        label TEXT NOT NULL,
                        confidence DOUBLE PRECISION,
                        face_detected BOOLEAN NOT NULL,
                        frame_width INTEGER,
                        frame_height INTEGER,
                        points_count INTEGER NOT NULL DEFAULT 0,
                        client_ip TEXT,
                        user_agent TEXT
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_engagement_predictions_created_at
                    ON engagement_predictions (created_at DESC);
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_engagement_predictions_label
                    ON engagement_predictions (label);
                    """
                )
            conn.commit()

        _set_last_error(None)
        return True
    except Exception as error:
        _set_last_error(error)
        return False


@contextmanager
def get_connection():
    if _connection_pool is None:
        init_db()

    conn = _connection_pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        _connection_pool.putconn(conn)


def should_save_prediction():
    global _prediction_counter
    with _counter_lock:
        _prediction_counter += 1
        return _prediction_counter % SAVE_EVERY_N_PREDICTIONS == 0


def save_prediction(label, confidence, face_detected, frame_width, frame_height, points_count, client_ip, user_agent):
    if not is_enabled() or not should_save_prediction():
        return False

    try:
        if _connection_pool is None:
            init_db()

        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO engagement_predictions (
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
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (
                        datetime.now(timezone.utc),
                        label,
                        confidence,
                        face_detected,
                        frame_width,
                        frame_height,
                        points_count,
                        client_ip,
                        user_agent,
                    ),
                )
            conn.commit()
        _set_last_error(None)
        return True
    except Exception as error:
        _set_last_error(error)
        return False
