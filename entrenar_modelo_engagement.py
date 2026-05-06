"""
entrenar_modelo_engagement.py
Proyecto 2 - Medidor de Engagement Facial

Entrena modelos supervisados usando dataset_engagement.csv y exporta el mejor
modelo para la demo web/webcam.
"""

from pathlib import Path
import shutil

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / "dataset_engagement.csv"
MODEL_FILE = BASE_DIR / "modelo_engagement.pkl"
SCALER_FILE = BASE_DIR / "scaler_engagement.pkl"
FEATURES_FILE = BASE_DIR / "columnas_features.pkl"

RANDOM_STATE = 42


def balancear_por_submuestreo(df, target_col):
    menor_clase = df[target_col].value_counts().min()
    partes = []
    for _, grupo in df.groupby(target_col):
        partes.append(grupo.sample(n=menor_clase, random_state=RANDOM_STATE))
    return pd.concat(partes).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def evaluar_modelo(nombre, pipeline, x_train, x_test, y_train, y_test):
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict(x_test)
    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
    )
    return {
        "nombre": nombre,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "cv_f1_macro": cv_scores.mean(),
        "reporte": classification_report(y_test, pred),
    }


def main():
    df = pd.read_csv(DATASET_FILE)
    columnas_features = [col for col in df.columns if col != "engagement"]

    df_balanceado = balancear_por_submuestreo(df, "engagement")
    x = df_balanceado[columnas_features]
    y = df_balanceado["engagement"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    modelos = [
        (
            "SVM RBF C=10 balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=10,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "SVM RBF C=30 balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=30,
                            gamma="scale",
                            class_weight="balanced",
                            probability=True,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Gradient Boosting",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=150,
                            learning_rate=0.1,
                            max_depth=4,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest balanced",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=None,
                            min_samples_leaf=2,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
    ]

    resultados = []
    for nombre, pipeline in modelos:
        print(f"\nEntrenando {nombre}...")
        resultado = evaluar_modelo(nombre, pipeline, x_train, x_test, y_train, y_test)
        resultados.append(resultado)
        print(
            f"{nombre}: accuracy={resultado['accuracy']:.4f}, "
            f"f1_macro={resultado['f1_macro']:.4f}, "
            f"f1_weighted={resultado['f1_weighted']:.4f}, "
            f"cv_f1_macro={resultado['cv_f1_macro']:.4f}"
        )
        print(resultado["reporte"])

    mejor = max(resultados, key=lambda item: item["f1_macro"])
    print(f"\nMejor modelo: {mejor['nombre']} (F1 macro={mejor['f1_macro']:.4f})")

    if MODEL_FILE.exists():
        shutil.copy2(MODEL_FILE, BASE_DIR / "modelo_engagement_backup.pkl")
    if SCALER_FILE.exists():
        shutil.copy2(SCALER_FILE, BASE_DIR / "scaler_engagement_backup.pkl")

    mejor_pipeline = mejor["pipeline"]
    joblib.dump(mejor_pipeline.named_steps["model"], MODEL_FILE)
    joblib.dump(mejor_pipeline.named_steps["scaler"], SCALER_FILE)
    joblib.dump(columnas_features, FEATURES_FILE)

    resumen = pd.DataFrame(
        [
            {
                "modelo": item["nombre"],
                "accuracy": item["accuracy"],
                "f1_macro": item["f1_macro"],
                "f1_weighted": item["f1_weighted"],
                "cv_f1_macro": item["cv_f1_macro"],
            }
            for item in resultados
        ]
    ).sort_values("f1_macro", ascending=False)
    resumen.to_csv(BASE_DIR / "metricas_modelos.csv", index=False)
    print("\nResumen guardado en metricas_modelos.csv")


if __name__ == "__main__":
    main()
