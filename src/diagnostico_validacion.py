"""TP Final 2026 | Fase 3, Paso 94: diagnóstico del LASSO en Validation.

Ejecutar: python src/diagnostico_validacion.py

- No lee observaciones de Test (nrows=736+157).
- No reoptimiza hiperparámetros: utiliza LASSO alpha=300 ya elegido.
- Ajusta StandardScaler únicamente con Train, dentro de un Pipeline.
- Guarda figuras, predicciones de Validation y tablas de diagnóstico.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite guardar figuras en ejecución desde terminal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "data" / "processed" / "dataset_final.csv"
CHECKSUMS = RAIZ / "data" / "checksums.json"
RESULTADOS = RAIZ / "results"
FIGURAS = RESULTADOS / "figures"
SEMILLA = 42
N_TRAIN = 736
N_VAL = 157
ALPHA = 300.0


def cargar_train_y_validation() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Verifica integridad y lee únicamente Train/Validation, sin cargar Test."""
    with CHECKSUMS.open(encoding="utf-8") as archivo:
        hash_esperado = json.load(archivo)["data/processed/dataset_final.csv"]
    contenido = DATASET.read_bytes()
    hash_obtenido = hashlib.sha256(contenido.replace(b"\r\n", b"\n")).hexdigest()
    if hash_obtenido != hash_esperado:
        raise ValueError("Checksum de dataset_final.csv inválido")

    datos = pd.read_csv(DATASET, nrows=N_TRAIN + N_VAL, parse_dates=["fecha"])
    if datos["particion"].tolist() != ["train"] * N_TRAIN + ["validation"] * N_VAL:
        raise ValueError("Orden de las particiones inesperado")
    if datos["fecha"].duplicated().any() or not datos["fecha"].is_monotonic_increasing:
        raise ValueError("Fechas duplicadas o desordenadas en Train/Validation")

    metadata = {"particion", "fecha", "fecha_siguiente", "archivo_origen", "target_manana"}
    caracteristicas = [c for c in datos.columns if c not in metadata]
    if len(caracteristicas) != 22:
        raise ValueError(f"Se esperaban 22 variables, hay {len(caracteristicas)}")
    if datos[caracteristicas + ["target_manana"]].isna().any().any():
        raise ValueError("Existen valores faltantes en las particiones de desarrollo")

    train = datos.iloc[:N_TRAIN].copy()
    validation = datos.iloc[N_TRAIN:].copy()
    return train, validation, caracteristicas


def graficar_predicciones(reales: np.ndarray, pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(reales, pred, s=22, alpha=0.65)
    extremo_inferior = float(min(np.min(reales), np.min(pred)))
    extremo_superior = float(max(np.max(reales), np.max(pred)))
    ax.plot([extremo_inferior, extremo_superior],
            [extremo_inferior, extremo_superior], linestyle="--", linewidth=1.5)
    ax.set(xlabel="Frío real de D+1 (kW)", ylabel="Frío predicho de D+1 (kW)",
           title="LASSO: valores reales vs. predicciones (Validation)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURAS / "01_predicho_vs_real_validation.png", dpi=160)
    plt.close(fig)


def graficar_residuos(pred: np.ndarray, residuos: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred, residuos, s=22, alpha=0.65)
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Frío predicho (kW)", ylabel="Error: real - predicho (kW)",
           title="LASSO: residuos vs. predicción (Validation)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURAS / "02_residuos_vs_prediccion.png", dpi=160)
    plt.close(fig)


def graficar_residuos_en_tiempo(fechas: pd.Series, residuos: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fechas, residuos, marker=".", linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Fecha de la predicción (día D)", ylabel="Error: real - predicho (kW)",
           title="LASSO: evolución temporal de los residuos (Validation)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURAS / "03_residuos_temporales.png", dpi=160)
    plt.close(fig)


def graficar_distribucion_residuos(residuos: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuos, bins=22, edgecolor="white", alpha=0.85)
    ax.axvline(0, linestyle="--", linewidth=1.5)
    ax.axvline(float(np.mean(residuos)), linestyle=":", linewidth=1.5,
               label=f"Sesgo medio = {np.mean(residuos):.1f}")
    ax.set(xlabel="Error: real - predicho (kW)", ylabel="Frecuencia",
           title="LASSO: distribución de los residuos (Validation)")
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURAS / "04_distribucion_residuos.png", dpi=160)
    plt.close(fig)


def graficar_coeficientes(coefs: pd.DataFrame) -> None:
    principales = coefs.nlargest(15, "magnitud").sort_values("coef_estandarizado")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(principales["caracteristica"], principales["coef_estandarizado"])
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set(xlabel="Coeficiente LASSO (variables estandarizadas)",
           title="15 características con mayor magnitud de coeficiente")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURAS / "05_coeficientes_lasso.png", dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    train, val, features = cargar_train_y_validation()

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", Lasso(alpha=ALPHA, max_iter=30000, random_state=SEMILLA)),
    ])
    modelo.fit(train[features], train["target_manana"])
    real = val["target_manana"].to_numpy(dtype=float)
    pred = np.asarray(modelo.predict(val[features]), dtype=float)
    residuos = real - pred
    errores_abs = np.abs(residuos)

    mae = mean_absolute_error(real, pred)
    rmse = mean_squared_error(real, pred) ** 0.5
    r2 = r2_score(real, pred)
    sesgo = float(residuos.mean())

    print("\nPASO 94 - DIAGNÓSTICO LASSO EN VALIDATION")
    print("Train:", train[features].shape, "| Validation:", val[features].shape)
    print("Test: no se cargó ni se utilizó")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")
    print(f"Sesgo medio (real - predicho): {sesgo:.2f} kW")
    print(f"Mediana del error absoluto: {np.median(errores_abs):.2f} kW")
    print(f"Percentil 90 del error absoluto: {np.percentile(errores_abs, 90):.2f} kW")
    print(f"Error absoluto máximo: {np.max(errores_abs):.2f} kW")
    print("Residuales positivos (subestimaciones):", int((residuos > 0).sum()))
    print("Residuales negativos (sobreestimaciones):", int((residuos < 0).sum()))

    tabla = pd.DataFrame({
        "fecha_D": val["fecha"].dt.strftime("%Y-%m-%d").to_numpy(),
        "fecha_D_mas_1": pd.to_datetime(val["fecha_siguiente"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "frio_real_D_mas_1": real,
        "frio_predicho_D_mas_1": pred,
        "residuo_real_menos_predicho": residuos,
        "error_absoluto": errores_abs,
    })
    tabla.to_csv(RESULTADOS / "lasso_validation_predicciones.csv", index=False)

    coefs = pd.DataFrame({
        "caracteristica": features,
        "coef_estandarizado": modelo.named_steps["lasso"].coef_,
    })
    coefs["magnitud"] = coefs["coef_estandarizado"].abs()
    coefs.sort_values("magnitud", ascending=False).to_csv(
        RESULTADOS / "lasso_coeficientes.csv", index=False
    )

    print("\nTOP 10 COEFICIENTES ESTANDARIZADOS POR MAGNITUD")
    print(coefs.nlargest(10, "magnitud").round(2).to_string(index=False))
    print("\nCINCO MAYORES ERRORES ABSOLUTOS EN VALIDATION")
    print(tabla.nlargest(5, "error_absoluto").round(2).to_string(index=False))

    graficar_predicciones(real, pred)
    graficar_residuos(pred, residuos)
    graficar_residuos_en_tiempo(val["fecha"], residuos)
    graficar_distribucion_residuos(residuos)
    graficar_coeficientes(coefs)
    print("\nSalidas en results/:")
    print("- figures/ (5 gráficos PNG)")
    print("- lasso_validation_predicciones.csv")
    print("- lasso_coeficientes.csv")
    print("\nImportancia: los coeficientes muestran asociación predictiva, no causalidad.")
    print("No se modificaron los hiperparámetros ni se evaluó Test.")


if __name__ == "__main__":
    main()
