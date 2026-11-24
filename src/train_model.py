"""TP Final 2026 | Fase 3, pasos 90-95: modelos, Optuna y registro final.

Guardá este archivo como src/train_model.py.
No lee filas de Test para modelar ni utiliza sus etiquetas.
Los estimadores lineales se escalan dentro de Pipeline ajustado solo con Train.
"""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset_final.csv"
CHECKSUMS = RAIZ / "data" / "checksums.json"
LOGS = RAIZ / "results" / "experiment_logs.csv"
SEMILLA = 42
N_TRAIN = 736
N_VALIDATION = 157
MODELOS_DIR = RAIZ / "models"
REGISTRO = MODELOS_DIR / "model_registry.json"
VERSION = "v1.0.0"


def verificar_integridad() -> None:
    """Verifica el hash registrado, tolerando la conversión CRLF→LF de Git/Windows.

    La normalización se hace solo para comparar el hash: no reescribe el CSV.
    """
    with CHECKSUMS.open(encoding="utf-8") as f:
        checksums = json.load(f)

    clave = "data/processed/dataset_final.csv"
    esperado = checksums[clave]
    contenido = DATASET.read_bytes()
    obtenido = hashlib.sha256(contenido.replace(b"\r\n", b"\n")).hexdigest()

    if obtenido != esperado:
        raise ValueError(
            "El SHA-256 del dataset no coincide con checksums.json "
            "(tampoco tras normalizar los saltos de línea)."
        )
    print("Integridad del dataset: OK")
    print("SHA-256 normalizado:", obtenido)


def cargar_desarrollo() -> tuple[pd.DataFrame, list[str]]:
    """Lee exclusivamente las primeras filas cronológicas de Train y Validation.

    El CSV de Fase 2 está ordenado y contiene Train, luego Validation y al final Test.
    nrows evita incluso cargar las filas de Test durante el desarrollo.
    """
    datos = pd.read_csv(DATASET, nrows=N_TRAIN + N_VALIDATION)
    datos["fecha"] = pd.to_datetime(datos["fecha"])

    secuencia = datos["particion"].tolist()
    if secuencia != ["train"] * N_TRAIN + ["validation"] * N_VALIDATION:
        raise ValueError(
            "El CSV no está organizado como Train → Validation → Test. "
            "Revisar la exportación del preprocesamiento."
        )

    if not datos["fecha"].is_monotonic_increasing or datos["fecha"].duplicated().any():
        raise ValueError("Las fechas de Train/Validation no están ordenadas o están duplicadas.")

    excluidas = {
        "particion", "fecha", "fecha_siguiente", "archivo_origen", "target_manana"
    }
    features = [col for col in datos.columns if col not in excluidas]

    if len(features) != 22:
        raise ValueError(f"Se esperaban 22 características, hay {len(features)}.")
    if datos[features + ["target_manana"]].isna().any().any():
        raise ValueError("Hay faltantes en Train/Validation.")
    if not np.isfinite(datos[features].to_numpy(dtype=float)).all():
        raise ValueError("Hay valores no finitos en Train/Validation.")

    return datos, features


def calcular_baselines(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice D+1 usando la información disponible hasta el cierre de D.

    La ventana de la media móvil es [D-6, ..., D]. Si faltan fechas
    calendario se calcula la media de los días observados (min_periods=1).
    """
    serie = datos.set_index("fecha")["frio_kw"].sort_index().asfreq("D")
    media_7 = serie.rolling(window=7, min_periods=1).mean()
    conteo_7 = serie.notna().rolling(window=7, min_periods=1).sum()

    validacion = datos.loc[datos["particion"].eq("validation")].copy()
    validacion["pred_persistencia"] = validacion["frio_kw"]
    validacion["pred_media_movil_7d"] = media_7.reindex(validacion["fecha"]).to_numpy()
    validacion["observados_en_ventana"] = conteo_7.reindex(validacion["fecha"]).to_numpy()

    registros = []
    for nombre, columna, parametros in [
        ("Persistencia", "pred_persistencia", {"regla": "consumo_dia_D"}),
        (
            "Media movil 7 dias", "pred_media_movil_7d",
            {"ventana_dias_calendario": 7, "min_periods": 1, "incluye_dia_D": True},
        ),
    ]:
        real = validacion["target_manana"]
        pred = validacion[columna]
        registros.append({
            "modelo": nombre,
            "particion": "validation",
            "filas": len(validacion),
            "MAE": mean_absolute_error(real, pred),
            "RMSE": mean_squared_error(real, pred) ** 0.5,
            "R2": r2_score(real, pred),
            "hiperparametros": json.dumps(parametros, ensure_ascii=False, sort_keys=True),
        })

    tabla = pd.DataFrame(registros).sort_values("MAE").reset_index(drop=True)
    return validacion, tabla


def entrenar_lineales(
    datos: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """Busca Ridge y LASSO; selecciona cada alpha por MAE de Validation."""
    train = datos.loc[datos["particion"].eq("train")]
    val = datos.loc[datos["particion"].eq("validation")]

    X_train = train[features]
    y_train = train["target_manana"]
    X_val = val[features]
    y_val = val["target_manana"]

    candidatos = {
        "Ridge": [0.1, 1, 10, 100, 1000, 10000],
        "LASSO": [10, 30, 100, 300, 1000],
    }
    registros = []

    for nombre, alphas in candidatos.items():
        for alpha in alphas:
            estimador = (
                Ridge(alpha=alpha, random_state=SEMILLA)
                if nombre == "Ridge"
                else Lasso(alpha=alpha, max_iter=30000, random_state=SEMILLA)
            )
            modelo = Pipeline([
                ("scaler", StandardScaler()),
                ("regresor", estimador),
            ])
            modelo.fit(X_train, y_train)
            pred = modelo.predict(X_val)

            parametros = {
                "alpha": alpha,
                "scaler": "StandardScaler",
                "features": len(features),
                "random_state": SEMILLA,
            }
            if nombre == "LASSO":
                parametros["max_iter"] = 30000

            registros.append({
                "modelo": nombre,
                "particion": "validation",
                "filas": len(val),
                "MAE": mean_absolute_error(y_val, pred),
                "RMSE": mean_squared_error(y_val, pred) ** 0.5,
                "R2": r2_score(y_val, pred),
                "hiperparametros": json.dumps(parametros, ensure_ascii=False, sort_keys=True),
                "alpha": alpha,
            })

    return pd.DataFrame(registros)


def optimizar_random_forest_optuna(
    datos: pd.DataFrame, features: list[str], n_trials: int = 25
) -> tuple[pd.DataFrame, dict]:
    """Ajusta Random Forest minimizando MAE de Validation con Optuna.

    El ajuste de hiperparámetros utiliza exclusivamente Train y Validation.
    Test NO se carga ni se consulta. Cada trial queda registrado para trazabilidad.
    """
    train = datos.loc[datos["particion"].eq("train")]
    val = datos.loc[datos["particion"].eq("validation")]

    X_train, y_train = train[features], train["target_manana"]
    X_val, y_val = val[features], val["target_manana"]

    def objetivo(trial: optuna.trial.Trial) -> float:
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_categorical(
                "max_depth", [None, 6, 10, 15, 20]
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 16, step=2),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0, step=0.2),
        }
        modelo = RandomForestRegressor(
            **parametros,
            random_state=SEMILLA,
            n_jobs=-1,
        )
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_val)

        mae = float(mean_absolute_error(y_val, pred))
        trial.set_user_attr("RMSE", float(mean_squared_error(y_val, pred) ** 0.5))
        trial.set_user_attr("R2", float(r2_score(y_val, pred)))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
        study_name="rf_validation_mae_paso91",
    )
    estudio.optimize(objetivo, n_trials=n_trials, n_jobs=1)

    registros = []
    for trial in estudio.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        configuracion = {
            **trial.params,
            "random_state": SEMILLA,
            "n_jobs": -1,
            "features": len(features),
            "optuna_trial": trial.number,
            "sampler": "TPESampler",
            "optimizacion": "MAE_validation",
        }
        registros.append({
            "modelo": "Random Forest (Optuna)",
            "particion": "validation",
            "filas": len(val),
            "MAE": trial.value,
            "RMSE": trial.user_attrs["RMSE"],
            "R2": trial.user_attrs["R2"],
            "hiperparametros": json.dumps(
                configuracion, ensure_ascii=False, sort_keys=True
            ),
            "trial": trial.number,
        })

    tabla = pd.DataFrame(registros)
    if tabla.empty:
        raise RuntimeError("Optuna no completó ningún ensayo de Random Forest.")
    return tabla, dict(estudio.best_trial.params)


def optimizar_xgboost_optuna(
    datos: pd.DataFrame, features: list[str], n_trials: int = 30
) -> tuple[pd.DataFrame, dict]:
    """Optimiza XGBoost exclusivamente con Train y Validation.

    Se minimiza el MAE de Validation. No se consulta Test ni
    se utiliza early stopping para evitar reglas adicionales
    de selección durante esta primera búsqueda acotada.
    """
    train = datos.loc[datos["particion"].eq("train")]
    val = datos.loc[datos["particion"].eq("validation")]

    X_train, y_train = train[features], train["target_manana"]
    X_val, y_val = val[features], val["target_manana"]

    def objetivo(trial: optuna.trial.Trial) -> float:
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 650, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.02, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.1
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        modelo = XGBRegressor(
            **parametros,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEMILLA,
            n_jobs=2,
            verbosity=0,
        )
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_val)

        mae = float(mean_absolute_error(y_val, pred))
        trial.set_user_attr("RMSE", float(mean_squared_error(y_val, pred) ** 0.5))
        trial.set_user_attr("R2", float(r2_score(y_val, pred)))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
        study_name="xgboost_validation_mae_paso92",
    )
    estudio.optimize(objetivo, n_trials=n_trials, n_jobs=1)

    registros = []
    for trial in estudio.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        configuracion = {
            **trial.params,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": SEMILLA,
            "n_jobs": 2,
            "features": len(features),
            "optuna_trial": trial.number,
            "sampler": "TPESampler",
            "optimizacion": "MAE_validation",
        }
        registros.append({
            "modelo": "XGBoost (Optuna)",
            "particion": "validation",
            "filas": len(val),
            "MAE": trial.value,
            "RMSE": trial.user_attrs["RMSE"],
            "R2": trial.user_attrs["R2"],
            "hiperparametros": json.dumps(
                configuracion, ensure_ascii=False, sort_keys=True
            ),
            "trial": trial.number,
        })

    tabla = pd.DataFrame(registros)
    if tabla.empty:
        raise RuntimeError("Optuna no completó ningún ensayo de XGBoost.")
    return tabla, dict(estudio.best_trial.params)


def optimizar_lightgbm_optuna(
    datos: pd.DataFrame, features: list[str], n_trials: int = 30
) -> tuple[pd.DataFrame, dict]:
    """Busca hiperparámetros de LightGBM minimizando MAE de Validation.

    Ajusta solo con Train, usa Validation para seleccionar y no consulta Test.
    La semilla 42 y opciones deterministas mejoran reproducibilidad en CPU.
    """
    train = datos.loc[datos["particion"].eq("train")]
    val = datos.loc[datos["particion"].eq("validation")]

    X_train, y_train = train[features], train["target_manana"]
    X_val, y_val = val[features], val["target_manana"]

    def objetivo(trial: optuna.trial.Trial) -> float:
        parametros = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 650, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31, 63]),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 35, step=5),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.1
            ),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        modelo = LGBMRegressor(
            **parametros,
            objective="regression",
            subsample_freq=1,  # activa el muestreo por filas (subsample)
            random_state=SEMILLA,
            n_jobs=2,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_val)

        mae = float(mean_absolute_error(y_val, pred))
        trial.set_user_attr("RMSE", float(mean_squared_error(y_val, pred) ** 0.5))
        trial.set_user_attr("R2", float(r2_score(y_val, pred)))
        return mae

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEMILLA),
        study_name="lightgbm_validation_mae_paso93",
    )
    estudio.optimize(objetivo, n_trials=n_trials, n_jobs=1)

    registros = []
    for trial in estudio.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        configuracion = {
            **trial.params,
            "objective": "regression",
            "subsample_freq": 1,
            "random_state": SEMILLA,
            "n_jobs": 2,
            "deterministic": True,
            "force_col_wise": True,
            "features": len(features),
            "optuna_trial": trial.number,
            "sampler": "TPESampler",
            "optimizacion": "MAE_validation",
        }
        registros.append({
            "modelo": "LightGBM (Optuna)",
            "particion": "validation",
            "filas": len(val),
            "MAE": trial.value,
            "RMSE": trial.user_attrs["RMSE"],
            "R2": trial.user_attrs["R2"],
            "hiperparametros": json.dumps(
                configuracion, ensure_ascii=False, sort_keys=True
            ),
            "trial": trial.number,
        })

    tabla = pd.DataFrame(registros)
    if tabla.empty:
        raise RuntimeError("Optuna no completó ningún ensayo de LightGBM.")
    return tabla, dict(estudio.best_trial.params)


def guardar_logs(tabla: pd.DataFrame) -> None:
    """Añade cada ejecución al tracking; mantiene el esquema de los logs existentes."""
    LOGS.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "no_disponible"

    columnas = [
        "modelo", "particion", "filas", "MAE", "RMSE", "R2", "hiperparametros"
    ]
    a_guardar = tabla[columnas].copy()
    a_guardar.insert(
        0, "timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    a_guardar["semilla"] = SEMILLA
    a_guardar["git_commit"] = commit
    a_guardar.to_csv(LOGS, mode="a", index=False, header=not LOGS.exists())




def verificar_codigo_versionado() -> str:
    """Exige que el código usado para entrenar corresponda a un commit real.

    El artefacto y el registro se incorporarán en un commit posterior;
    `git_commit` representa la versión de código/datos con que se entrenó.
    """
    archivos_criticos = [
        "src/train_model.py",
        "requirements.txt",
        "data/processed/dataset_final.csv",
        "data/checksums.json",
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL,
        ).strip()
        estado = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *archivos_criticos],
            cwd=RAIZ, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("No se puede obtener el commit de Git para el registro.") from exc
    if estado:
        raise RuntimeError(
            "Hay cambios sin commitear en archivos necesarios para reproducir "
            "el entrenamiento:\n" + estado +
            "\nHacé primero un commit con src/train_model.py, "
            "requirements.txt y cualquier cambio de datos."
        )
    return commit


def cargar_ganador_de_logs() -> tuple[float, dict]:
    """Comprueba en el tracking que el menor MAE de Validation es LASSO.

    Los logs existentes contienen distintos ensayos y pueden repetir
    experimentos; el criterio sigue siendo MAE sobre Validation.
    """
    if not LOGS.exists():
        raise FileNotFoundError(
            f"Falta {LOGS}. Ejecutá primero el entrenamiento/Optuna."
        )
    logs = pd.read_csv(LOGS)
    requeridas = {"modelo", "particion", "MAE", "hiperparametros"}
    if not requeridas.issubset(logs.columns):
        raise ValueError(f"El log no contiene las columnas: {sorted(requeridas)}")

    dev = logs.loc[logs["particion"].eq("validation")].copy()
    dev["MAE"] = pd.to_numeric(dev["MAE"], errors="coerce")
    dev = dev.dropna(subset=["MAE"])
    if dev.empty:
        raise ValueError("No hay métricas válidas de Validation en el log.")

    mejor = dev.loc[dev["MAE"].idxmin()]
    if mejor["modelo"] != "LASSO":
        raise ValueError(
            f"En experiment_logs.csv el menor MAE lo tiene {mejor['modelo']} "
            f"({mejor['MAE']:.2f}), no LASSO. Revisar selección antes de guardar."
        )
    parametros = json.loads(mejor["hiperparametros"])
    alpha = float(parametros["alpha"])
    print("Ganador identificado en el tracking:", mejor["modelo"])
    print(f"MAE registrado en Validation: {mejor['MAE']:.2f}")
    print("Alpha seleccionado:", alpha)
    return alpha, {"MAE_log": float(mejor["MAE"]), "parametros_log": parametros}


def finalizar_modelo() -> None:
    """Evalúa con Train→Validation y reajusta una sola vez con Train+Val.

    No carga ni inspecciona Test. Tampoco repite las búsquedas con Optuna.
    Guarda el Pipeline completo (StandardScaler + Lasso) en joblib.
    """
    print("\nPASO 95 - ENTRENAMIENTO FINAL Y MODEL REGISTRY")
    commit = verificar_codigo_versionado()
    verificar_integridad()
    alpha, info_log = cargar_ganador_de_logs()
    datos, features = cargar_desarrollo()  # pd.read_csv(nrows=893)

    train = datos.loc[datos["particion"].eq("train")].copy()
    val = datos.loc[datos["particion"].eq("validation")].copy()

    def nuevo_pipeline() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regresor", Lasso(alpha=alpha, max_iter=30000, random_state=SEMILLA)),
        ])

    # Métricas reproducidas con el ajuste en Train únicamente.
    modelo_validacion = nuevo_pipeline()
    modelo_validacion.fit(train[features], train["target_manana"])
    pred = modelo_validacion.predict(val[features])
    mae = float(mean_absolute_error(val["target_manana"], pred))
    rmse = float(mean_squared_error(val["target_manana"], pred) ** 0.5)
    r2 = float(r2_score(val["target_manana"], pred))

    if abs(mae - info_log["MAE_log"]) > 0.02:
        raise RuntimeError(
            f"MAE actual ({mae:.4f}) no coincide con el log "
            f"({info_log['MAE_log']:.4f}). Revisá entorno/datos antes de registrar."
        )

    print(f"Validation (modelo entrenado solo en Train): MAE={mae:.2f}, "
          f"RMSE={rmse:.2f}, R²={r2:.4f}")

    # Una vez fijado el modelo y alpha, usar todas las filas de desarrollo.
    final = nuevo_pipeline()
    final.fit(datos[features], datos["target_manana"])
    print("Reentrenamiento: Train + Validation =", len(datos), "filas")
    print("Shape:", datos[features].shape)
    print("Test: NO cargado, NO evaluado")

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_modelo = MODELOS_DIR / f"modelo_{VERSION}.pkl"
    if ruta_modelo.exists():
        raise FileExistsError(
            f"Ya existe {ruta_modelo}. No se sobrescriben modelos versionados. "
            "Si corresponde una nueva versión, cambiá VERSION."
        )

    # joblib serializa tanto el escalador como el regresor.
    joblib.dump(final, ruta_modelo)
    hash_modelo = hashlib.sha256(ruta_modelo.read_bytes()).hexdigest()
    with CHECKSUMS.open(encoding="utf-8") as f:
        checksum_datos = json.load(f)["data/processed/dataset_final.csv"]

    entrada = {
        "version": VERSION,
        "modelo": "LASSO",
        "path": ruta_modelo.relative_to(RAIZ).as_posix(),
        "fecha_entrenamiento_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_script": "src/train_model.py",
        "git_commit": commit,
        "semilla": SEMILLA,
        "hiperparametros": {
            "alpha": alpha,
            "max_iter": 30000,
            "random_state": SEMILLA,
            "escalado": "StandardScaler dentro de Pipeline",
        },
        "metricas_validation": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "filas": len(val),
            "nota": "Métricas previas al reentrenamiento, con ajuste solo en Train",
        },
        "reentrenamiento": {
            "particiones": ["train", "validation"],
            "filas": len(datos),
        },
        "features": features,
        "dataset": "data/processed/dataset_final.csv",
        "dataset_sha256_normalizado": checksum_datos,
        "modelo_sha256": hash_modelo,
    }
    if REGISTRO.exists():
        with REGISTRO.open(encoding="utf-8") as f:
            registro = json.load(f)
        if not isinstance(registro.get("models"), dict):
            raise ValueError("El model_registry existente no tiene formato 'models'.")
        if VERSION in registro["models"]:
            raise FileExistsError(f"La versión {VERSION} ya está en el registry.")
    else:
        registro = {"latest": VERSION, "models": {}}
    registro["models"][VERSION] = entrada
    registro["latest"] = VERSION

    with REGISTRO.open("w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    # Confirmar que el artefacto se puede deserializar y conserva el orden.
    cargado = joblib.load(ruta_modelo)
    if list(cargado.feature_names_in_) != features:
        raise RuntimeError("El orden de las features no coincide al cargar el modelo.")
    if not np.isfinite(cargado.predict(val[features])).all():
        raise RuntimeError("El modelo serializado produce predicciones no finitas.")

    print("\nARTEFACTOS GENERADOS")
    print("Modelo:", ruta_modelo.relative_to(RAIZ))
    print("Registry:", REGISTRO.relative_to(RAIZ))
    print("Versión:", VERSION)
    print("Commit de código usado para entrenar:", commit)
    print("SHA-256 del modelo:", hash_modelo)
    print("Modelo serializado y recargado: OK")
    print("No se reportaron métricas de Test.")

def main() -> None:
    verificar_integridad()
    datos, features = cargar_desarrollo()
    validacion, tabla_baselines = calcular_baselines(datos)

    print("\nFASE 3 - BASELINES EN VALIDATION")
    print("Filas de Train:", (datos["particion"] == "train").sum())
    print("Filas de Validation:", len(validacion))
    print("\n", tabla_baselines.drop(columns="hiperparametros").round(2).to_string(index=False))
    incompletas = int((validacion["observados_en_ventana"] < 7).sum())
    print(f"\nVentanas de 7 días con alguna fecha ausente: {incompletas}/{len(validacion)}")
    print("La media móvil promedia los días observados si faltan fechas.")

    tabla_lineales = entrenar_lineales(datos, features)
    print("\nPASO 90 - RIDGE Y LASSO (22 CARACTERÍSTICAS)")
    for nombre in ["Ridge", "LASSO"]:
        seleccion = tabla_lineales.loc[
            tabla_lineales["modelo"].eq(nombre),
            ["alpha", "MAE", "RMSE", "R2"],
        ].sort_values("MAE")
        print(f"\n{nombre} - candidatos ordenados por MAE")
        print(seleccion.round(2).to_string(index=False))
        ganador = seleccion.iloc[0]
        print(f"Mejor alpha de {nombre} por MAE de Validation: {ganador['alpha']:g}")

    mejores_lineales = (
        tabla_lineales.loc[tabla_lineales.groupby("modelo")["MAE"].idxmin()]
        .drop(columns="alpha")
    )
    comparacion = pd.concat(
        [tabla_baselines, mejores_lineales], ignore_index=True
    ).sort_values("MAE")
    print("\nCOMPARACIÓN PROVISIONAL - VALIDATION")
    print(comparacion.drop(columns="hiperparametros").round(2).to_string(index=False))
    print("\nComparación preliminar: continúa Random Forest con Optuna.")

    print("\nPASO 91 - RANDOM FOREST CON OPTUNA")
    print("Criterio: minimizar MAE en Validation | Ensayos: 25 | Semilla: 42")
    tabla_rf, mejores_parametros_rf = optimizar_random_forest_optuna(
        datos, features, n_trials=25
    )
    print("\nMejores 10 ensayos de Random Forest:")
    print(
        tabla_rf[["trial", "MAE", "RMSE", "R2"]]
        .sort_values("MAE")
        .head(10)
        .round(2)
        .to_string(index=False)
    )
    print("\nMejores hiperparámetros de RF:")
    print(json.dumps(mejores_parametros_rf, indent=2, ensure_ascii=False))

    mejor_rf = tabla_rf.loc[tabla_rf["MAE"].idxmin()].drop(labels=["trial"])
    comparacion_actualizada = pd.concat(
        [comparacion, mejor_rf.to_frame().T], ignore_index=True
    ).sort_values("MAE")
    print("\nCOMPARACIÓN ACTUALIZADA - VALIDATION")
    print(
        comparacion_actualizada
        .drop(columns="hiperparametros")
        .round(2)
        .to_string(index=False)
    )
    print("\nPASO 92 - XGBOOST CON OPTUNA")
    print("Criterio: minimizar MAE en Validation | Ensayos: 30 | Semilla: 42")
    tabla_xgb, mejores_parametros_xgb = optimizar_xgboost_optuna(
        datos, features, n_trials=30
    )
    print("\nMejores 10 ensayos de XGBoost:")
    print(
        tabla_xgb[["trial", "MAE", "RMSE", "R2"]]
        .sort_values("MAE")
        .head(10)
        .round(2)
        .to_string(index=False)
    )
    print("\nMejores hiperparámetros de XGBoost:")
    print(json.dumps(mejores_parametros_xgb, indent=2, ensure_ascii=False))

    mejor_xgb = tabla_xgb.loc[tabla_xgb["MAE"].idxmin()].drop(labels=["trial"])
    comparacion_final = pd.concat(
        [comparacion_actualizada, mejor_xgb.to_frame().T], ignore_index=True
    ).sort_values("MAE")
    print("\nCOMPARACIÓN ACTUALIZADA - VALIDATION")
    print(
        comparacion_final.drop(columns="hiperparametros")
        .round(2).to_string(index=False)
    )
    print("\nPASO 93 - LIGHTGBM CON OPTUNA")
    print("Criterio: minimizar MAE en Validation | Ensayos: 30 | Semilla: 42")
    tabla_lgbm, mejores_parametros_lgbm = optimizar_lightgbm_optuna(
        datos, features, n_trials=30
    )
    print("\nMejores 10 ensayos de LightGBM:")
    print(
        tabla_lgbm[["trial", "MAE", "RMSE", "R2"]]
        .sort_values("MAE")
        .head(10)
        .round(2)
        .to_string(index=False)
    )
    print("\nMejores hiperparámetros de LightGBM:")
    print(json.dumps(mejores_parametros_lgbm, indent=2, ensure_ascii=False))

    mejor_lgbm = tabla_lgbm.loc[
        tabla_lgbm["MAE"].idxmin()
    ].drop(labels=["trial"])
    comparacion_completa = pd.concat(
        [comparacion_final, mejor_lgbm.to_frame().T], ignore_index=True
    ).sort_values("MAE")
    print("\nCOMPARACIÓN COMPLETA - VALIDATION")
    print(
        comparacion_completa.drop(columns="hiperparametros")
        .round(2).to_string(index=False)
    )
    print("\nTest permanece sin consultar.")

    guardar_logs(tabla_baselines)
    guardar_logs(tabla_lineales)
    guardar_logs(tabla_rf)
    guardar_logs(tabla_xgb)
    guardar_logs(tabla_lgbm)
    print("\nExperimentos registrados en:", LOGS.relative_to(RAIZ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelado Fase 3 - TP final")
    parser.add_argument(
        "--finalizar", action="store_true",
        help="Reentrena el mejor modelo en Train+Validation y crea models/ sin repetir Optuna",
    )
    args = parser.parse_args()
    if args.finalizar:
        finalizar_modelo()
    else:
        main()
