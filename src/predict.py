"""Fase 4: predicción diaria con el modelo versionado.

Reutiliza las transformaciones de preprocessing_pipeline.py y los
parámetros originales de Train registrados en data_lineage.json.
Sin usar etiquetas de Test ni ajustar parámetros con el Excel nuevo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# Importar funciones ya usadas en Fase 2. No duplicar su normalización temporal.
from preprocessing_pipeline import (
    limpiar_temporal, seleccionar_jornadas_temporales, agregar_calendario,
    normalizar_horas, evaluar_calidad_frio,
    aplicar_umbrales_extremos, aplicar_imputacion_industrial,
    calcular_ratios_industriales, calcular_historicas, incorporar_historicas,
    aplicar_imputacion_historica, aplicar_transformacion_derivada
)

RAIZ = Path(__file__).resolve().parent.parent
REGISTRO = RAIZ / "models" / "model_registry.json"
LINAJE = RAIZ / "data" / "processed" / "data_lineage.json"

# Columnas efectivamente utilizadas para construir las 22 características
# seleccionadas. Columnas/hojas adicionales del Excel son permitidas.
COLUMNAS_REQUERIDAS = {
    "Consolidado EE": [
        "DIA", "HORA", "Frio (Kw)", "Servicios (Kw)",
        "Planta (Kw)", "Envasado (Kw)", "Linea 4 (Kw)",
    ],
    "Consolidado Produccion": [
        "DIA", "HORA", "Hl Producido Bodega",
    ],
    "Consolidado Agua": [
        "DIA", "HORA", "Agua Calderas",
    ],
    "Consolidado GasVapor": [
        "DIA", "HORA", "Tot Vap Lav L3 / Hora",
        "Vapor Cocina (Kg)", "Tot_Vapor_CIP_Bodega",
        "VAPOR DE LINEA 4 KG", "Vapor Envasado (Kg)",
        "Vapor Servicio (Kg)",
    ],
    "Consolidado Aire": [
        "DIA", "HORA", "Totaliador_Aire_L4",
        "Totalizador_Aire_Bodega",
    ],
}


def _sha256(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def load_model(version: str = "latest"):
    """Carga una versión del modelo desde el registry y verifica su hash."""
    if not REGISTRO.is_file():
        raise FileNotFoundError(
            f"No se encontró el registro de modelos: {REGISTRO}"
        )

    with REGISTRO.open(encoding="utf-8") as f:
        registro = json.load(f)

    if version == "latest":
        version = registro.get("latest")

    versiones = registro.get("models", {})
    if version not in versiones:
        raise ValueError(
            f"Versión '{version}' no registrada. "
            f"Versiones disponibles: {', '.join(sorted(versiones))}"
        )

    info = versiones[version]
    ruta_modelo = RAIZ / info["path"]
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            f"Falta el modelo registrado '{version}': {ruta_modelo}"
        )

    hash_esperado = info.get("modelo_sha256")
    if hash_esperado and _sha256(ruta_modelo) != hash_esperado:
        raise ValueError(
            f"El SHA-256 del modelo '{version}' no coincide con el registry."
        )

    # joblib solo debe cargar modelos propios/de procedencia confiable.
    modelo = joblib.load(ruta_modelo)
    if not callable(getattr(modelo, "predict", None)):
        raise TypeError("El archivo registrado no contiene un estimador predictivo.")

    columnas_esperadas = info.get("features")
    if not columnas_esperadas:
        raise ValueError("El registro del modelo no contiene 'features'.")

    nombres_modelo = getattr(modelo, "feature_names_in_", None)
    if nombres_modelo is not None and list(nombres_modelo) != list(columnas_esperadas):
        raise ValueError(
            "Las características del modelo serializado no coinciden "
            "con el orden registrado en model_registry.json."
        )

    return modelo, columnas_esperadas, version


def validar_entrada_excel(filepath: str | Path) -> dict[str, list[str]]:
    """Verifica hojas/columnas por nombre, sin asumir posiciones.

    Permite hojas y columnas adicionales. Reporta claramente lo que falta.
    Esta función no examina objetivos, ni aprende parámetros del Excel.
    """
    archivo = Path(filepath)
    if not archivo.is_file():
        raise FileNotFoundError(f"No existe el archivo de entrada: {archivo}")
    if archivo.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("El archivo de entrada debe ser Excel (.xlsx o .xlsm).")

    with pd.ExcelFile(archivo, engine="openpyxl") as libro:
        hojas_faltantes = sorted(set(COLUMNAS_REQUERIDAS) - set(libro.sheet_names))
        if hojas_faltantes:
            raise ValueError(
                "Faltan hojas necesarias en el Excel: "
                + ", ".join(hojas_faltantes)
                + ". Hojas disponibles: "
                + ", ".join(libro.sheet_names)
            )

        columnas_por_hoja = {}
        for hoja, requeridas in COLUMNAS_REQUERIDAS.items():
            columnas = [str(c).strip() for c in pd.read_excel(
                libro, sheet_name=hoja, nrows=0
            ).columns]
            if len(columnas) != len(set(columnas)):
                raise ValueError(f"Hay nombres de columnas duplicados en '{hoja}'.")
            faltantes = sorted(set(requeridas) - set(columnas))
            if faltantes:
                raise ValueError(
                    f"En la hoja '{hoja}' faltan las columnas: "
                    + ", ".join(faltantes)
                )
            columnas_por_hoja[hoja] = columnas

    return columnas_por_hoja



def _columnas_originales(libro: pd.ExcelFile, hoja: str, requeridas: list[str]):
    """Devuelve las cabeceras originales, respetando nombres y no posiciones."""
    originales = list(pd.read_excel(libro, sheet_name=hoja, nrows=0).columns)
    nombres = [str(c).strip() for c in originales]
    if len(nombres) != len(set(nombres)):
        raise ValueError(f"Hay columnas repetidas después de quitar espacios en '{hoja}'.")
    mapa = dict(zip(nombres, originales))
    faltantes = sorted(set(requeridas) - set(mapa))
    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en la hoja '{hoja}': {', '.join(faltantes)}"
        )
    return [mapa[c] for c in requeridas]


def _normalizar_horas(horas: pd.Series) -> pd.Series:
    """Aplica el mismo ajuste de +1 segundo que la Fase 2."""
    return normalizar_horas(pd.to_timedelta(horas.astype(str), errors="coerce"))


def _leer_hoja_necesaria(libro: pd.ExcelFile, hoja: str, requeridas: list[str]):
    cabeceras = _columnas_originales(libro, hoja, requeridas)
    datos = pd.read_excel(libro, sheet_name=hoja, usecols=cabeceras)
    datos.columns = datos.columns.astype(str).str.strip()
    datos["fecha"] = pd.to_datetime(datos["DIA"], errors="coerce").dt.normalize()
    datos["hora_normalizada"] = _normalizar_horas(datos["HORA"])
    for col in requeridas:
        if col not in {"DIA", "HORA"}:
            datos[col] = pd.to_numeric(datos[col], errors="coerce")
    return datos


def _cierres_completos(datos: pd.DataFrame, columnas: list[str], hoja: str):
    """Cierres solo de jornadas con exactamente 25 horarios válidos."""
    cero = pd.Timedelta(0)
    cierre = pd.Timedelta(hours=23, minutes=59)
    esperadas = {pd.Timedelta(hours=h) for h in range(24)} | {cierre}
    base = datos.dropna(subset=["fecha", "hora_normalizada"]).copy()
    # Si una fecha tiene más de 25 filas, es ambigua, aunque sus horarios repitan.
    controles = base.groupby("fecha")["hora_normalizada"].agg(
        lecturas="size", horarios="nunique", esquema=lambda x: set(x) == esperadas
    )
    fechas_aprobadas = controles.index[
        (controles.lecturas == 25) & (controles.horarios == 25) & controles.esquema
    ]
    cierres = base.loc[
        base.fecha.isin(fechas_aprobadas) & base.hora_normalizada.eq(cierre),
        ["fecha"] + columnas,
    ].copy()
    if cierres.fecha.duplicated().any():
        raise ValueError(f"Cierres diarios duplicados en '{hoja}'.")
    return cierres.sort_values("fecha").reset_index(drop=True), len(controles), int((controles.lecturas > 25).sum())


def auditar_jornadas_excel(
    filepath: str | Path, *, devolver_cierres: bool = False
) -> dict | tuple[dict, dict]:
    """Audita el Excel sin entrenar ni predecir ni consultar etiquetas D+1.

    Utiliza exactamente la limpieza y selección temporal de EE de Fase 2.
    El resto de áreas comparte el control de 25 lecturas y cierre a 23:59.
    """
    archivo = Path(filepath)
    validar_entrada_excel(archivo)
    resumen = {}
    cierres_area = {}
    with pd.ExcelFile(archivo, engine="openpyxl") as libro:
        # La lógica de EE sí puede reutilizarse sin modificaciones de la Fase 2.
        ee = _leer_hoja_necesaria(libro, "Consolidado EE", COLUMNAS_REQUERIDAS["Consolidado EE"])
        ee_raw = ee.copy()
        ee_raw["archivo_origen"] = archivo.name
        ee_raw["fila_excel"] = np.arange(2, len(ee_raw) + 2)
        ee_raw["hora"] = pd.to_timedelta(ee_raw["HORA"].astype(str), errors="coerce")
        ee_raw["frio_kw"] = ee_raw["Frio (Kw)"]
        ee_temporal = seleccionar_jornadas_temporales(limpiar_temporal(ee_raw))
        # Calidad del totalizador: mismas seis reglas de Fase 2, sin los
        # asserts de conteos fijos del dataset histórico.
        calidad, _ = evaluar_calidad_frio(ee_temporal)
        apta = calidad["apta_calidad"]
        fechas_frio = pd.Index(calidad.index.get_level_values("fecha")[apta])
        if fechas_frio.empty:
            raise ValueError("No se encontraron jornadas válidas del totalizador de frío.")
        resumen["Consolidado EE"] = {
            "jornadas_completas": len(calidad), "jornadas_frio_validas": int(apta.sum()),
            "jornadas_frio_descartadas_calidad": int((~apta).sum()),
        }
        cierres_ee, _, _ = _cierres_completos(ee_temporal, [
            "Frio (Kw)", "Servicios (Kw)", "Planta (Kw)",
            "Envasado (Kw)", "Linea 4 (Kw)",
        ], "Consolidado EE")
        cierres_area["Consolidado EE"] = cierres_ee.loc[
            cierres_ee.fecha.isin(fechas_frio)
        ].reset_index(drop=True)

        for hoja, requeridas in COLUMNAS_REQUERIDAS.items():
            if hoja == "Consolidado EE":
                continue
            datos = _leer_hoja_necesaria(libro, hoja, requeridas)
            columnas = [c for c in requeridas if c not in {"DIA", "HORA"}]
            cierres, jornadas, apiladas = _cierres_completos(datos, columnas, hoja)
            cierres_area[hoja] = cierres
            resumen[hoja] = {
                "fechas_en_archivo": jornadas,
                "fechas_apiladas": apiladas,
                "jornadas_completas": len(cierres),
            }
    # La Fase 2 conservó días de EE aunque faltaran áreas; no perderlos aquí.
    fechas_base = cierres_area["Consolidado EE"].fecha
    resumen["dias_predecibles"] = int(len(fechas_base))
    resumen["fechas_inicio_fin"] = [
        fechas_base.min().strftime("%Y-%m-%d"),
        fechas_base.max().strftime("%Y-%m-%d"),
    ]
    for hoja, cierres in cierres_area.items():
        if hoja != "Consolidado EE":
            resumen[hoja]["dias_ee_sin_cierre_area"] = int(
                (~fechas_base.isin(cierres.fecha)).sum()
            )
    if devolver_cierres:
        return resumen, cierres_area
    return resumen


def _mostrar_auditoria(resumen: dict) -> None:
    print("\nPASO 115 - AUDITORÍA DE JORNADAS")
    for hoja in COLUMNAS_REQUERIDAS:
        print(f"{hoja}: {json.dumps(resumen[hoja], ensure_ascii=False)}")
    print("Días D con frío válido:", resumen["dias_predecibles"])
    print("Período:", " a ".join(resumen["fechas_inicio_fin"]))
    print("No se utilizaron targets ni se generaron predicciones.")



# Corresponde a las columnas industriales efectivamente utilizadas por el
# modelo, más el denominador necesario para construir el indicador del ratio.
# No es necesario exigir hojas Mycom o KPI porque ninguna de sus variables
# fue seleccionada por LASSO.
NOMBRES_MODELO_POR_HOJA = {
    "Consolidado EE": {
        "Frio (Kw)": "frio_kw",
        "Servicios (Kw)": "EE | Servicios (Kw)",
        "Planta (Kw)": "EE | Planta (Kw)",
        "Envasado (Kw)": "EE | Envasado (Kw)",
        "Linea 4 (Kw)": "EE | Linea 4 (Kw)",
    },
    "Consolidado Produccion": {
        "Hl Producido Bodega": "Produccion | Hl Producido Bodega",
    },
    "Consolidado Agua": {
        "Agua Calderas": "Agua | Agua Calderas",
    },
    "Consolidado GasVapor": {
        "Tot Vap Lav L3 / Hora": "GasVapor | Tot Vap Lav L3 / Hora",
        "Vapor Cocina (Kg)": "GasVapor | Vapor Cocina (Kg)",
        "Tot_Vapor_CIP_Bodega": "GasVapor | Tot_Vapor_CIP_Bodega",
        "VAPOR DE LINEA 4 KG": "GasVapor | VAPOR DE LINEA 4 KG",
        "Vapor Envasado (Kg)": "GasVapor | Vapor Envasado (Kg)",
        "Vapor Servicio (Kg)": "GasVapor | Vapor Servicio (Kg)",
    },
    "Consolidado Aire": {
        "Totaliador_Aire_L4": "Aire | Totaliador_Aire_L4",
        "Totalizador_Aire_Bodega": "Aire | Totalizador_Aire_Bodega",
    },
}


def cargar_reglas_entrenamiento(features: list[str]) -> dict:
    """Recupera medianas y umbrales aprendidos en la Fase 2, sin recalcularlos."""
    if not LINAJE.is_file():
        raise FileNotFoundError(f"No se encontró el linaje de Fase 2: {LINAJE}")
    with LINAJE.open(encoding="utf-8") as f:
        linaje = json.load(f)
    previstas = linaje.get("dataset_final", {}).get("caracteristicas")
    if previstas != features:
        raise ValueError(
            "Las características del dataset de Fase 2 no coinciden con "
            "model_registry.json (incluido su orden)."
        )
    reglas = linaje.get("parametros_train", {})
    esperadas = (
        "medianas_industriales", "umbrales_industriales",
        "medianas_historicas", "medianas_derivadas",
        "umbrales_derivadas", "indicadores_derivadas",
    )
    faltantes = [clave for clave in esperadas if clave not in reglas]
    if faltantes:
        raise ValueError(f"Faltan parámetros de Train en data_lineage.json: {faltantes}")
    return reglas


def construir_caracteristicas_predictivas(
    filepath: str | Path, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Construye X(D) sin utilizar nunca el valor del objetivo D+1.

    Reutiliza las transformaciones compartidas de Fase 2. Para nuevas
    fuentes, los lags solo pueden utilizar el historial disponible en el Excel.
    Los extremos, las imputaciones y los indicadores respetan Train.
    """
    reglas = cargar_reglas_entrenamiento(features)
    auditoria, cierres = auditar_jornadas_excel(filepath, devolver_cierres=True)

    # Mantener todas las jornadas con Frío válido, aunque falten otras áreas.
    base = cierres["Consolidado EE"].rename(
        columns=NOMBRES_MODELO_POR_HOJA["Consolidado EE"]
    ).copy()
    for hoja in (
        "Consolidado Produccion", "Consolidado Agua",
        "Consolidado GasVapor", "Consolidado Aire",
    ):
        mapa = NOMBRES_MODELO_POR_HOJA[hoja]
        parte = cierres[hoja][["fecha", *mapa]].rename(columns=mapa)
        base = base.merge(parte, how="left", on="fecha", validate="one_to_one")

    base = base.sort_values("fecha").reset_index(drop=True)
    fechas = base["fecha"].copy()
    columnas_industriales = [
        col for mapa in NOMBRES_MODELO_POR_HOJA.values() for col in mapa.values()
    ]
    X_original = base[columnas_industriales].apply(pd.to_numeric, errors="coerce")

    # Aplicar límites aprendidos exclusivamente en Train.
    limites = pd.Series(reglas["umbrales_industriales"], dtype=float)
    X_limpio, _ = aplicar_umbrales_extremos(X_original, limites)

    # Calcular los ratios necesarios ANTES de imputar las variables base.
    r_serv = "Ratio | Frio sobre Servicios"
    r_bodega = "Ratio | Frio por Hl Bodega"
    nombres_ratios = [r_serv, r_bodega]
    ratios = calcular_ratios_industriales(X_limpio, nombres=nombres_ratios)
    if r_bodega not in reglas["indicadores_derivadas"]:
        raise ValueError("Train no registró el indicador faltante del ratio Bodega.")
    medianas_derivadas = reglas["medianas_derivadas"]
    sin_mediana = sorted(set(nombres_ratios) - set(medianas_derivadas))
    if sin_mediana:
        raise ValueError(f"Faltan medianas derivadas de Train: {sin_mediana}")
    ratios = aplicar_transformacion_derivada(
        ratios, nombres_ratios, medianas_derivadas,
        reglas["umbrales_derivadas"],
        [c for c in reglas["indicadores_derivadas"] if c in nombres_ratios]
    )

    # El modelo final no utiliza indicadores industriales de faltantes.
    medianas = pd.Series(reglas["medianas_industriales"], dtype=float)
    sin_mediana = sorted(set(columnas_industriales) - set(medianas.index))
    if sin_mediana:
        raise ValueError(f"Faltan medianas industriales: {sin_mediana}")
    X = aplicar_imputacion_industrial(X_limpio, medianas)
    X = agregar_calendario(X, fechas)

    # Utilizar la serie de frío sin imputar, como en el entrenamiento.
    columnas_hist = ["frio_lag_7d", "frio_media_3d", "frio_media_7d"]
    medianas_hist = reglas["medianas_historicas"]
    for columna in columnas_hist:
        if columna not in medianas_hist:
            raise ValueError(f"Falta mediana histórica de Train: {columna}")
    X = incorporar_historicas(X, fechas, calcular_historicas(base))
    X = aplicar_imputacion_historica(X, medianas_hist, columnas_hist)

    # Los indicadores de ratios se calcularon antes de imputarlos.
    X = pd.concat([X, ratios], axis=1)

    faltantes = sorted(set(features) - set(X.columns))
    if faltantes:
        raise ValueError(f"No se pudieron reconstruir las características: {faltantes}")
    X = X.loc[:, features].copy()  # orden idéntico al del modelo registrado
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("La matriz de predicción contiene NaN o infinitos.")
    if X.columns.tolist() != features or len(X) != len(fechas):
        raise ValueError("Dimensiones/orden de características inconsistentes.")
    return X, fechas, auditoria


def _mostrar_caracteristicas(X: pd.DataFrame, fechas: pd.Series, auditoria: dict):
    print("\nPASO 116 - CARACTERÍSTICAS PARA PREDICCIÓN")
    print("Días D con frío válido:", len(fechas))
    print("Dimensiones X:", X.shape)
    print("Faltantes finales:", int(X.isna().sum().sum()))
    print("Fechas:", fechas.min().date(), "a", fechas.max().date())
    print("Indicadores de ausencia (suma):")
    for columna in X.columns:
        if columna.endswith("| faltante"):
            print(f" - {columna}: {int(X[columna].sum())}")
    print("Muestra (3 fechas, sin etiquetas):")
    columnas_ejemplo = ["frio_kw", "cos_anual", "frio_lag_7d", "frio_media_3d"]
    print(pd.concat([fechas.rename("fecha_D"), X[columnas_ejemplo]], axis=1).head(3).to_string(index=False))
    print("Reglas de mediana/umbrales recuperadas de data_lineage.json: OK")
    print("No se generaron predicciones y no se utilizaron targets.")



def comparar_con_train(
    filepath: str | Path, X: pd.DataFrame, fechas: pd.Series, features: list[str]
) -> None:
    """Audita equivalencia de features con el CSV consolidado SOLO en Train.

    No accede a la columna target_manana ni consulta particiones Validation/Test.
    Las variables históricas se contrastan solo cuando siete días previos
    están presentes en Train y provienen de este mismo Excel; así evitamos
    falsos errores causados por ventanas parciales o solapamientos de fuentes.
    """
    ruta_csv = RAIZ / "data" / "processed" / "dataset_final.csv"
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No existe el dataset histórico de Fase 2: {ruta_csv}"
        )

    # Deliberadamente NO solicitamos ni leemos target_manana.
    columnas = ["fecha", "archivo_origen", "particion", *features]
    referencia = pd.read_csv(
        ruta_csv, usecols=columnas, parse_dates=["fecha"]
    )
    referencia = referencia.loc[
        referencia["particion"].eq("train")
        & referencia["archivo_origen"].eq(Path(filepath).name)
    ].copy()

    if referencia.empty:
        raise ValueError(
            "No hay filas de Train cuyo archivo_origen coincida con el Excel "
            "recibido. Probá con una de las fuentes históricas utilizadas."
        )
    if referencia["fecha"].duplicated().any():
        raise ValueError("Train contiene fechas duplicadas para la fuente.")

    nuevas = X.copy().reset_index(drop=True)
    nuevas.insert(0, "fecha", pd.to_datetime(fechas).reset_index(drop=True))
    if nuevas["fecha"].duplicated().any():
        raise ValueError("Las características reconstruidas tienen fechas duplicadas.")

    referencia = referencia.set_index("fecha").sort_index()
    nuevas = nuevas.set_index("fecha").sort_index()
    comunes = referencia.index.intersection(nuevas.index).sort_values()
    if comunes.empty:
        raise ValueError("No hay fechas comunes entre Train y el Excel de entrada.")

    # No usar etiquetas; solo comparar características.
    industriales_y_derivadas = [
        c for c in features if c not in {
            "frio_lag_7d", "frio_media_3d", "frio_media_7d",
            "frio_media_7d | faltante"
        }
    ]
    historicas = [c for c in features if c not in industriales_y_derivadas]

    # Para los lags, exigimos 7 días anteriores de Train y de la MISMA fuente.
    # Si no se cumple, Fase 2 pudo utilizar valores de otros Excel o
    # fechas fuera del subconjunto supervisado exportado en dataset_final.
    fechas_train_fuente = set(referencia.index)
    contexto_completo = pd.DatetimeIndex([
        dia for dia in comunes
        if all(
            (dia - pd.Timedelta(days=desfase)) in fechas_train_fuente
            for desfase in range(1, 8)
        )
    ])

    def verificar(grupo: list[str], fechas_comparables: pd.DatetimeIndex, titulo: str):
        print(f"\n{titulo}: {len(fechas_comparables)} fechas")
        if len(fechas_comparables) == 0 or not grupo:
            print("Sin suficientes fechas comparables para esta sección.")
            return None
        esperado = referencia.loc[fechas_comparables, grupo].to_numpy(dtype=float)
        calculado = nuevas.loc[fechas_comparables, grupo].to_numpy(dtype=float)
        coincide = np.isclose(
            esperado, calculado, rtol=1e-8, atol=1e-5, equal_nan=True
        )
        errores = np.argwhere(~coincide)
        print("Características comparadas:", len(grupo))
        print("Celdas diferentes:", len(errores))
        if len(errores):
            print("Primeras diferencias (fecha | característica | Train | Predicción):")
            for i, j in errores[:12]:
                print(
                    f" - {fechas_comparables[i].date()} | {grupo[j]} | "
                    f"{esperado[i,j]:.8g} | {calculado[i,j]:.8g}"
                )
        return len(errores) == 0

    print("\nPASO 117 - COMPARACIÓN DE CARACTERÍSTICAS CON TRAIN")
    print("Fuente:", Path(filepath).name)
    print("Fechas de Train de esa fuente:", len(referencia))
    print("Fechas comunes reconstruidas:", len(comunes))
    resultado_actual = verificar(
        industriales_y_derivadas, comunes, "Características del día D"
    )
    resultado_hist = verificar(
        historicas, contexto_completo,
        "Características históricas con 7 días de contexto de la misma fuente"
    )
    if resultado_actual is False or resultado_hist is False:
        raise AssertionError(
            "Se detectaron diferencias respecto del preprocesamiento de Train. "
            "No generar predicciones hasta investigar las causas."
        )
    if resultado_hist is None:
        print(
            "AVISO: no hubo fechas con siete días previos comparables. "
            "La equivalencia de lags queda sin verificar."
        )
    print("Comparación completada sin consultar targets ni Test.")

def load_and_preprocess(
    filepath: str | Path, features: list[str] | None = None, *, version: str = "latest"
) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el Excel y devuelve X(D) y fechas mediante el pipeline compartido.

    Si no se proporcionan las columnas, usa el modelo registrado indicado.
    No ajusta parámetros ni usa el objetivo D+1.
    """
    if features is None:
        _, features, _ = load_model(version)
    X, fechas, _ = construir_caracteristicas_predictivas(filepath, features)
    return X, fechas


def predict_consumption(filepath: str | Path, version: str = "latest") -> pd.DataFrame:
    """Predice el consumo de D+1 para cada jornada D válida del Excel.

    La columna `fecha` identifica al día D, tal como lo exige la consigna.
    No se lee la columna objetivo ni se calculan métricas de evaluación.
    """
    modelo, features, version_cargada = load_model(version)
    X, fechas = load_and_preprocess(filepath, features)

    if X.empty:
        raise ValueError("El Excel no contiene jornadas válidas para predecir.")
    if X.columns.tolist() != list(features):
        raise ValueError("El orden de las características difiere del modelo registrado.")

    predicciones = np.asarray(modelo.predict(X), dtype=float)
    if predicciones.ndim != 1 or len(predicciones) != len(X):
        raise ValueError(
            "El modelo devolvió un número o una dimensión inesperada de predicciones."
        )
    if not np.isfinite(predicciones).all():
        raise ValueError("El modelo produjo predicciones NaN o infinitas.")

    salida = pd.DataFrame({
        "fecha": pd.to_datetime(fechas).dt.normalize().to_numpy(),
        "prediccion_frio_kw": predicciones,
    })
    salida = salida.sort_values("fecha").reset_index(drop=True)
    if salida["fecha"].duplicated().any():
        raise ValueError("Hay fechas duplicadas en las predicciones diarias.")

    print("\nPREDICCIÓN DIARIA - FASE 4")
    print("Modelo:", version_cargada)
    print("Características:", X.shape[1])
    print("Días predichos:", len(salida))
    print("Período D:", salida["fecha"].min().date(), "a", salida["fecha"].max().date())
    print("Fechas duplicadas:", int(salida["fecha"].duplicated().sum()))
    print("Predicciones no finitas: 0")
    if (predicciones < 0).any():
        print(
            "ADVERTENCIA: el modelo predijo consumos negativos en",
            int((predicciones < 0).sum()),
            "días. Se conservan para no alterar el modelo entrenado."
        )
    print("No se utilizaron etiquetas, y no se evaluó Test.")
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fase 4: predicción diaria y verificaciones de Excel/modelo registrado."
    )
    parser.add_argument("archivo_excel", type=Path)
    parser.add_argument("--version", default="latest")
    parser.add_argument(
        "--validar", action="store_true",
        help="Comprueba entrada y modelo; todavía no genera predicciones.",
    )
    parser.add_argument(
        "--auditar", action="store_true",
        help="Revisa los cierres diarios de las cinco hojas sin generar predicciones."
    )
    parser.add_argument(
        "--caracteristicas", action="store_true",
        help="Reconstruye las 22 características con reglas de Train, sin predecir."
    )
    parser.add_argument(
        "--comparar-train", action="store_true",
        help="Compara las 22 características reconstruidas con las de Train, sin etiquetas."
    )
    parser.add_argument(
        "--salida", type=Path, default=Path("results/predicciones.csv"),
        help="CSV de salida, por defecto results/predicciones.csv en el proyecto."
    )
    args = parser.parse_args()

    if args.comparar_train:
        _, features, version = load_model(args.version)
        print("Modelo cargado:", version, "| Características:", len(features))
        X, fechas, _ = construir_caracteristicas_predictivas(
            args.archivo_excel, features
        )
        comparar_con_train(args.archivo_excel, X, fechas, features)
        return

    if args.caracteristicas:
        _, features, version = load_model(args.version)
        print("Modelo cargado:", version, "| Características:", len(features))
        X, fechas, resumen = construir_caracteristicas_predictivas(
            args.archivo_excel, features
        )
        _mostrar_caracteristicas(X, fechas, resumen)
        return

    if args.auditar:
        hojas = validar_entrada_excel(args.archivo_excel)
        _, features, version = load_model(args.version)
        print("Modelo cargado:", version, "| Características:", len(features))
        _mostrar_auditoria(auditar_jornadas_excel(args.archivo_excel))
        return

    if args.validar:
        hojas = validar_entrada_excel(args.archivo_excel)
        _, features, version = load_model(args.version)
        print("VALIDACIÓN DE ENTRADA Y MODELO: OK")
        print("Archivo:", args.archivo_excel)
        print("Hojas necesarias verificadas:", len(hojas))
        print("Versión cargada:", version)
        print("Características registradas:", len(features))
        print("Todavía NO se generaron predicciones.")
        return

    # Modo por defecto: predicción, SIN leer datos objetivos.
    resultados = predict_consumption(args.archivo_excel, version=args.version)
    ruta_salida = args.salida if args.salida.is_absolute() else RAIZ / args.salida
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(
        ruta_salida, index=False, encoding="utf-8",
        date_format="%Y-%m-%d", lineterminator="\n",
    )
    print("\nPredicciones generadas en:", ruta_salida)
    print("Columnas del CSV:", resultados.columns.tolist())
    print("Muestra de predicciones:")
    print(resultados.head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
