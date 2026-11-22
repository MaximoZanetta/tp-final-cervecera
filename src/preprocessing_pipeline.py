
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from datetime import datetime, timezone

# Rutas del proyecto
RAIZ = Path(__file__).resolve().parent.parent
DATA_DIR = RAIZ / "data"
PROCESSED_DIR = DATA_DIR / "processed"

NOMBRES_ARCHIVOS = [
    "Totalizadores Planta 2020_2022.xlsx",
    "Totalizadores Planta - 2021_2023.xlsx",
    "Totalizadores Planta 2022_2023.xlsx",
]

ARCHIVOS = [
    DATA_DIR / nombre
    for nombre in NOMBRES_ARCHIVOS
]


def calcular_sha256(ruta):
    sha = hashlib.sha256()

    with ruta.open("rb") as f:
        for bloque in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha.update(bloque)

    return sha.hexdigest()


def verificar_fuentes():
    ruta_checksums = DATA_DIR / "checksums.json"

    with ruta_checksums.open(
        encoding="utf-8"
    ) as f:
        checksums = json.load(f)

    for archivo in ARCHIVOS:
        clave = f"data/{archivo.name}"

        if not archivo.is_file():
            raise FileNotFoundError(archivo)

        if calcular_sha256(archivo) != checksums[clave]:
            raise ValueError(
                f"Checksum incorrecto: {archivo.name}"
            )

    print("Archivos originales verificados:", len(ARCHIVOS))


def cargar_electricidad():
    partes = []

    for archivo in ARCHIVOS:
        df = pd.read_excel(
            archivo,
            sheet_name="Consolidado EE"
        )

        df.columns = df.columns.astype(str).str.strip()

        df["archivo_origen"] = archivo.name
        df["fila_excel"] = np.arange(2, len(df) + 2)

        partes.append(df)

    ee_raw = pd.concat(
        partes,
        ignore_index=True
    )

    ee_raw["fecha"] = pd.to_datetime(
        ee_raw["DIA"], errors="coerce"
    ).dt.normalize()

    ee_raw["hora"] = pd.to_timedelta(
        ee_raw["HORA"].astype(str),
        errors="coerce"
    )

    ee_raw["frio_kw"] = pd.to_numeric(
        ee_raw["Frio (Kw)"],
        errors="coerce"
    )

    return ee_raw


def limpiar_temporal(ee_raw):
    """
    Excluye registros sin fecha u hora y normaliza
    desfases exactos de un segundo.
    """

    # 1. Excluir registros sin referencia temporal
    ee = ee_raw.dropna(
        subset=["fecha", "hora"]
    ).copy()

    registros_excluidos = len(ee_raw) - len(ee)

    # 2. Definir los 25 horarios esperados
    horarios_esperados = pd.TimedeltaIndex(
        [pd.Timedelta(hours=h) for h in range(24)]
        + [pd.Timedelta(hours=23, minutes=59)]
    )

    un_segundo = pd.Timedelta(seconds=1)

    # 3. Conservar el horario original
    ee["hora_original"] = ee["hora"]

    # 4. Detectar desfases exactos
    desfase = (
        ee["hora_original"] - un_segundo
    ).isin(horarios_esperados)

    # 5. Normalizar sin alterar la lectura original
    ee["hora_normalizada"] = (
        ee["hora_original"].where(
            ~desfase,
            ee["hora_original"] - un_segundo
        )
    )

    # 6. Controlar las colisiones de horarios
    claves_originales = [
        "archivo_origen", "fecha", "hora_original"
    ]

    claves_normalizadas = [
        "archivo_origen", "fecha", "hora_normalizada"
    ]

    duplicados_antes = ee.duplicated(
        claves_originales
    ).sum()

    duplicados_despues = ee.duplicated(
        claves_normalizadas
    ).sum()

    print("\nLIMPIEZA TEMPORAL")
    print("Registros excluidos:", registros_excluidos)
    print("Registros conservados:", len(ee))
    print("Horarios normalizados:", int(desfase.sum()))
    print(
        "Duplicados nuevos:",
        int(duplicados_despues - duplicados_antes)
    )

    # Verificaciones de integridad
    assert ee["fecha"].notna().all()
    assert ee["hora_normalizada"].notna().all()
    assert len(ee) + registros_excluidos == len(ee_raw)

    return ee


def seleccionar_jornadas_temporales(ee_trabajo):
    """
    Excluye fechas con jornadas apiladas y conserva
    días con 25 horarios y cierre numérico.
    """

    claves = ["archivo_origen", "fecha"]

    # 1. Detectar fechas con más de 25 registros
    conteo = (
        ee_trabajo.groupby(claves)
        .size()
        .reset_index(name="registros")
    )

    fechas_apiladas = conteo.loc[
        conteo["registros"] > 25,
        claves
    ]

    fechas_ambiguas = set(fechas_apiladas["fecha"])

    # 2. Auditar los bloques horarios apilados
    apilados = (
        ee_trabajo.merge(
            fechas_apiladas,
            on=claves,
            how="inner"
        )
        .sort_values(claves + ["fila_excel"])
        .copy()
    )

    apilados["bloque"] = (
        apilados["hora_normalizada"]
        .eq(pd.Timedelta(0))
        .groupby([
            apilados["archivo_origen"],
            apilados["fecha"]
        ])
        .cumsum()
    )

    total_bloques = (
        apilados.groupby(claves + ["bloque"])
        .ngroups
    )

    # 3. Excluir fechas ambiguas de todos los Excel
    ee_sin_apilados = ee_trabajo.loc[
        ~ee_trabajo["fecha"].isin(fechas_ambiguas)
    ].copy()

    # 4. Definir los 25 horarios esperados
    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }

    cierre = pd.Timedelta(hours=23, minutes=59)
    horarios_validos.add(cierre)

    # 5. Verificar cada jornada por archivo y fecha
    control_temporal = (
        ee_sin_apilados.groupby(claves)
        .agg(
            n_registros=("hora_normalizada", "size"),
            n_horas=("hora_normalizada", "nunique"),
            esquema_correcto=(
                "hora_normalizada",
                lambda s: set(s) == horarios_validos
            )
        )
        .reset_index()
    )

    control_temporal["valida"] = (
        (control_temporal["n_registros"] == 25)
        & (control_temporal["n_horas"] == 25)
        & control_temporal["esquema_correcto"]
    )

    # 6. Recuperar las jornadas completas
    jornadas_validas = control_temporal.loc[
        control_temporal["valida"],
        claves
    ]

    ee_temporal = ee_sin_apilados.merge(
        jornadas_validas,
        on=claves,
        how="inner",
        validate="many_to_one"
    )

    # 7. Exigir un cierre numérico a las 23:59
    cierres = ee_temporal.loc[
        ee_temporal["hora_normalizada"] == cierre,
        claves + ["frio_kw"]
    ].dropna(subset=["frio_kw"])

    ee_temporal = ee_temporal.merge(
        cierres[claves],
        on=claves,
        how="inner",
        validate="many_to_one"
    )

    # 8. Mostrar los resultados
    print("\nCONTROL DE JORNADAS")

    print(
        "Combinaciones archivo-fecha apiladas:",
        len(fechas_apiladas)
    )

    print("Fechas ambiguas:", len(fechas_ambiguas))
    print("Bloques apilados identificados:", total_bloques)

    print(
        "Jornadas completas con cierre numérico:",
        len(cierres)
    )

    print(
        "Fechas distintas conservadas:",
        ee_temporal["fecha"].nunique()
    )

    print("Registros horarios:", len(ee_temporal))

    # 9. Comprobaciones de integridad
    assert len(ee_temporal) == len(cierres) * 25
    assert not cierres.duplicated(claves).any()

    return ee_temporal


def controlar_calidad_frio(ee_temporal):
    """
    Evalúa la calidad de las 25 lecturas de cada
    jornada y conserva cierres de fuentes válidas.
    """

    claves = ["archivo_origen", "fecha"]

    datos = ee_temporal.sort_values(
        claves + ["hora_normalizada"]
    ).copy()

    # Diferencias consecutivas dentro de cada jornada
    datos["variacion"] = (
        datos.groupby(claves)["frio_kw"].diff()
    )

    # Resumen por archivo y fecha
    calidad = datos.groupby(claves).agg(
        lecturas_nulas=(
            "frio_kw",
            lambda s: s.isna().sum()
        ),
        descensos=(
            "variacion",
            lambda s: (s < 0).sum()
        ),
        negativos=(
            "frio_kw",
            lambda s: (s < 0).sum()
        ),
        maximo=("frio_kw", "max")
    )

    inicio = pd.Timedelta(0)
    cierre = pd.Timedelta(hours=23, minutes=59)

    # Lecturas iniciales y finales
    calidad["inicio"] = (
        datos.loc[
            datos["hora_normalizada"] == inicio
        ]
        .set_index(claves)["frio_kw"]
    )

    calidad["cierre"] = (
        datos.loc[
            datos["hora_normalizada"] == cierre
        ]
        .set_index(claves)["frio_kw"]
    )

    # Identificar problemas de calidad
    calidad["reinicio_incorrecto"] = (
        calidad["inicio"].ne(0)
    )

    calidad["tiene_descensos"] = (
        calidad["descensos"] > 0
    )

    calidad["valor_extremo"] = (
        calidad["maximo"] > 1_000_000
    )

    calidad["cierre_invalido"] = (
        calidad["cierre"].isna()
        | (calidad["cierre"] <= 0)
    )

    calidad["tiene_negativos"] = (
        calidad["negativos"] > 0
    )

    calidad["tiene_nulos"] = (
        calidad["lecturas_nulas"] > 0
    )

    controles = [
        "reinicio_incorrecto",
        "tiene_descensos",
        "valor_extremo",
        "cierre_invalido",
        "tiene_negativos",
        "tiene_nulos"
    ]

    # Excluir jornadas con al menos un problema
    calidad["apta_calidad"] = (
        ~calidad[controles].any(axis=1)
    )

    # Recuperar los cierres de las jornadas aprobadas
    cierres_aprobados = (
        calidad.loc[
            calidad["apta_calidad"],
            ["cierre"]
        ]
        .reset_index()
        .rename(columns={"cierre": "frio_kw"})
    )

    # Fechas que perdieron todas sus fuentes
    fechas_originales = set(
        calidad.index.get_level_values("fecha")
    )

    fechas_aprobadas = set(
        cierres_aprobados["fecha"]
    )

    fechas_perdidas = (
        fechas_originales - fechas_aprobadas
    )

    # Resultados
    print("\nCONTROL DE CALIDAD DE FRÍO")

    print("Jornadas evaluadas:", len(calidad))

    for control in controles:
        print(
            f"{control}:",
            int(calidad[control].sum())
        )

    print(
        "\nJornadas excluidas:",
        int((~calidad["apta_calidad"]).sum())
    )

    print(
        "Jornadas aprobadas:",
        len(cierres_aprobados)
    )

    print(
        "Fechas distintas conservadas:",
        cierres_aprobados["fecha"].nunique()
    )

    print(
        "Fechas sin ninguna fuente válida:",
        len(fechas_perdidas)
    )

    print(
        "Fechas con múltiples fuentes válidas:",
        int(
            (
                cierres_aprobados.groupby("fecha")
                .size() > 1
            ).sum()
        )
    )

    # Verificaciones
    assert len(calidad) == 1626
    assert len(cierres_aprobados) == 1589
    assert cierres_aprobados["fecha"].nunique() == 1127

    return cierres_aprobados, calidad


def consolidar_cierres(cierres_aprobados):
    """
    Resuelve fechas repetidas entre archivos.

    - Excluye fechas con cierres contradictorios.
    - Prioriza el Excel más reciente cuando coinciden.
    - Conserva una observación por fecha.
    """

    datos = cierres_aprobados.copy()

    # 1. Detectar cierres diferentes para una misma fecha
    cantidad_valores = (
        datos.groupby("fecha")["frio_kw"].nunique()
    )

    fechas_conflictivas = cantidad_valores[
        cantidad_valores > 1
    ].index

    conflictos = datos.loc[
        datos["fecha"].isin(fechas_conflictivas)
    ].sort_values(["fecha", "archivo_origen"])

    print("\nCONSOLIDACIÓN DE FUENTES")
    print(
        "Fechas con cierres contradictorios:",
        len(fechas_conflictivas)
    )

    if not conflictos.empty:
        print("\nDetalle de conflictos:")
        print(conflictos.to_string(index=False))

    # 2. Excluir la fecha de TODAS las fuentes
    datos = datos.loc[
        ~datos["fecha"].isin(fechas_conflictivas)
    ].copy()

    # 3. Priorizar los archivos más recientes
    # NOMBRES_ARCHIVOS está ordenado de antiguo a reciente
    prioridad = {
        nombre: i
        for i, nombre in enumerate(NOMBRES_ARCHIVOS)
    }

    datos["prioridad_fuente"] = (
        datos["archivo_origen"].map(prioridad)
    )

    if datos["prioridad_fuente"].isna().any():
        raise ValueError("Se encontró una fuente desconocida.")

    # 4. Elegir la fuente de mayor prioridad
    datos = datos.sort_values(
        ["fecha", "prioridad_fuente"],
        ascending=[True, False]
    )

    frio_diario = (
        datos.drop_duplicates(
            subset="fecha",
            keep="first"
        )
        [["fecha", "archivo_origen", "frio_kw"]]
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    # 5. Mostrar resultados
    print(
        "\nJornadas aprobadas antes:",
        len(cierres_aprobados)
    )
    print(
        "Fechas consolidadas:",
        len(frio_diario)
    )
    print(
        "Fechas duplicadas restantes:",
        frio_diario["fecha"].duplicated().sum()
    )

    print("\nFuentes seleccionadas:")
    print(
        frio_diario["archivo_origen"].value_counts()
    )

    # 6. Validaciones
    assert len(fechas_conflictivas) == 1
    assert len(frio_diario) == 1126
    assert frio_diario["fecha"].is_unique

    return frio_diario, conflictos

def construir_objetivo(frio_diario):
    """
    Construye el consumo de refrigeración del día
    siguiente solo para fechas consecutivas.
    """

    datos = (
        frio_diario
        .sort_values("fecha")
        .reset_index(drop=True)
        .copy()
    )

    # Siguiente fecha observada en los datos
    datos["fecha_siguiente"] = (
        datos["fecha"].shift(-1)
    )

    # Verificar que sea exactamente el día siguiente
    consecutivo = (
        datos["fecha_siguiente"]
        == datos["fecha"] + pd.Timedelta(days=1)
    )

    # Consumo de la siguiente observación
    consumo_siguiente = datos["frio_kw"].shift(-1)

    # Crear objetivo solo si las fechas son consecutivas
    datos["target_manana"] = (
        consumo_siguiente.where(consecutivo)
    )

    # Resumen
    print("\nCONSTRUCCIÓN DE VARIABLE OBJETIVO")
    print("Fechas disponibles:", len(datos))

    print(
        "Pares consecutivos D -> D+1:",
        int(consecutivo.sum())
    )

    print(
        "Fechas sin objetivo válido:",
        int(datos["target_manana"].isna().sum())
    )

    print(
        "Primer día:",
        datos["fecha"].min().date()
    )

    print(
        "Último día:",
        datos["fecha"].max().date()
    )

    # Comprobaciones
    assert len(datos) == 1126
    assert datos["target_manana"].notna().sum() == 1054
    assert datos["target_manana"].isna().sum() == 72

    return datos


def integrar_electricidad(dataset_base, ee_temporal):
    """
    Integra los cierres eléctricos del día D usando
    la fecha y fuente seleccionadas para Frío.
    """

    claves = ["archivo_origen", "fecha"]
    cierre = pd.Timedelta(hours=23, minutes=59)

    # 1. Recuperar cierres de electricidad
    ee_cierres = ee_temporal.loc[
        ee_temporal["hora_normalizada"] == cierre
    ].copy()

    # Una sola lectura por fuente y fecha
    assert not ee_cierres.duplicated(claves).any()

    # 2. Seleccionar las variables eléctricas
    # Excluir Frío, que ya está en dataset_base
    columnas_ee = [
        c for c in ee_cierres.columns
        if c.endswith("(Kw)") and c != "Frio (Kw)"
    ]

    electricidad = ee_cierres[
        claves + columnas_ee
    ].copy()

    # 3. Convertir a valores numéricos
    for columna in columnas_ee:
        electricidad[columna] = pd.to_numeric(
            electricidad[columna],
            errors="coerce"
        )

    # 4. Identificar el área de cada característica
    electricidad = electricidad.rename(
        columns={
            c: f"EE | {c}"
            for c in columnas_ee
        }
    )

    # 5. Integrar por fecha y archivo
    dataset_integrado = dataset_base.merge(
        electricidad,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    # 6. Mostrar resultados
    print("\nINTEGRACIÓN DE ELECTRICIDAD")

    print("Fechas:", len(dataset_integrado))
    print("Variables añadidas:", len(columnas_ee))
    print("Columnas totales:", dataset_integrado.shape[1])

    print(
        "Pares D -> D+1:",
        dataset_integrado["target_manana"].notna().sum()
    )

    columnas_agregadas = [
        f"EE | {c}" for c in columnas_ee
    ]

    print("\nVariables eléctricas con más faltantes:")

    print(
        dataset_integrado[columnas_agregadas]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # 7. Verificaciones
    assert len(dataset_integrado) == 1126
    assert len(columnas_ee) == 17
    assert dataset_integrado.shape[1] == 22
    assert dataset_integrado["fecha"].is_unique

    assert (
        dataset_integrado["target_manana"].notna().sum()
        == 1054
    )

    return dataset_integrado


def integrar_produccion(dataset_integrado, dataset_base):
    """
    Integra las variables de producción del día D,
    respetando la fecha y el archivo de origen.
    """

    claves = ["archivo_origen", "fecha"]
    cierre = pd.Timedelta(hours=23, minutes=59)

    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }
    horarios_validos.add(cierre)

    partes_produccion = []

    # 1. Leer las hojas de producción
    for archivo in ARCHIVOS:

        df = pd.read_excel(
            archivo,
            sheet_name="Consolidado Produccion"
        )

        df.columns = df.columns.astype(str).str.strip()

        df["archivo_origen"] = archivo.name

        df["fecha"] = pd.to_datetime(
            df["DIA"], errors="coerce"
        ).dt.normalize()

        horas = pd.to_timedelta(
            df["HORA"].astype(str),
            errors="coerce"
        )

        # 2. Normalizar desfases exactos de un segundo
        un_segundo = pd.Timedelta(seconds=1)

        desfase = (
            horas - un_segundo
        ).isin(horarios_validos)

        df["hora_normalizada"] = horas.where(
            ~desfase,
            horas - un_segundo
        )

        # 3. Utilizar solo fechas elegidas de esta fuente
        fechas_elegidas = dataset_base.loc[
            dataset_base["archivo_origen"] == archivo.name,
            "fecha"
        ]

        df = df.loc[
            df["fecha"].isin(fechas_elegidas)
        ].copy()

        # 4. Eliminar variable duplicada en el EDA
        df = df.drop(
            columns=["Hl de Mosto Copia"],
            errors="ignore"
        )

        partes_produccion.append(df)

    produccion_raw = pd.concat(
        partes_produccion,
        ignore_index=True
    )

    # 5. Identificar jornadas completas
    control = (
        produccion_raw.groupby(claves)["hora_normalizada"]
        .agg(
            registros="size",
            horas_distintas="nunique",
            esquema=lambda s: set(s) == horarios_validos
        )
    )

    completas = control.loc[
        (control["registros"] == 25)
        & (control["horas_distintas"] == 25)
        & control["esquema"]
    ].reset_index()[claves]

    # 6. Recuperar cierres de las jornadas completas
    produccion_cierres = produccion_raw.merge(
        completas,
        on=claves,
        how="inner",
        validate="many_to_one"
    )

    produccion_cierres = produccion_cierres.loc[
        produccion_cierres["hora_normalizada"] == cierre
    ].copy()

    # 7. Seleccionar variables de producción
    excluir = {
        "DIA", "HORA", "Id", "archivo_origen",
        "fecha", "hora_normalizada"
    }

    columnas_prod = [
        c for c in produccion_cierres.columns
        if c not in excluir
        and not c.startswith("Unnamed")
    ]

    produccion_cierres = produccion_cierres[
        claves + columnas_prod
    ].copy()

    for columna in columnas_prod:
        produccion_cierres[columna] = pd.to_numeric(
            produccion_cierres[columna],
            errors="coerce"
        )

    # 8. Identificar el área de cada variable
    produccion_cierres = produccion_cierres.rename(
        columns={
            c: f"Produccion | {c}"
            for c in columnas_prod
        }
    )

    assert not produccion_cierres.duplicated(claves).any()

    # 9. Integrar sin perder fechas
    resultado = dataset_integrado.merge(
        produccion_cierres,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    columnas_agregadas = [
        f"Produccion | {c}"
        for c in columnas_prod
    ]

    print("\nINTEGRACIÓN DE PRODUCCIÓN")
    print("Fechas:", len(resultado))
    print("Variables añadidas:", len(columnas_prod))
    print("Columnas totales:", resultado.shape[1])

    print(
        "Pares D -> D+1:",
        resultado["target_manana"].notna().sum()
    )

    print(
        "Fechas sin cierre de producción:",
        resultado[columnas_agregadas].isna().all(axis=1).sum()
    )

    print("\nFaltantes por variable:")
    print(
        resultado[columnas_agregadas]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    # 10. Verificaciones contra el notebook
    assert len(resultado) == 1126
    assert len(columnas_prod) == 10
    assert resultado.shape[1] == 32
    assert resultado["fecha"].is_unique
    assert resultado["target_manana"].notna().sum() == 1054

    return resultado



def integrar_mycom(dataset_integrado, dataset_base):
    """
    Integra los siete compresores Mycom del día D,
    respetando fecha, fuente y cierre de las 23:59.
    """

    claves = ["archivo_origen", "fecha"]
    cierre = pd.Timedelta(hours=23, minutes=59)

    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }
    horarios_validos.add(cierre)

    columnas_mycom = [
        f"KW Mycom {i}" for i in range(1, 8)
    ]

    partes_mycom = []

    # 1. Leer la hoja de totalizadores de cada Excel
    for archivo in ARCHIVOS:

        df = pd.read_excel(
            archivo,
            sheet_name="Totalizadores Energia"
        )

        df.columns = df.columns.astype(str).str.strip()

        # Verificar columnas necesarias
        necesarias = ["DIA", "HORA"] + columnas_mycom

        faltantes = [
            c for c in necesarias if c not in df.columns
        ]

        if faltantes:
            raise ValueError(
                f"Faltan columnas en {archivo.name}: {faltantes}"
            )

        df["archivo_origen"] = archivo.name

        df["fecha"] = pd.to_datetime(
            df["DIA"], errors="coerce"
        ).dt.normalize()

        horas = pd.to_timedelta(
            df["HORA"].astype(str),
            errors="coerce"
        )

        # 2. Corregir desfases exactos de un segundo
        un_segundo = pd.Timedelta(seconds=1)

        desfase = (
            horas - un_segundo
        ).isin(horarios_validos)

        df["hora_normalizada"] = horas.where(
            ~desfase,
            horas - un_segundo
        )

        # 3. Mantener las fechas elegidas de este Excel
        fechas_elegidas = dataset_base.loc[
            dataset_base["archivo_origen"] == archivo.name,
            "fecha"
        ]

        df = df.loc[
            df["fecha"].isin(fechas_elegidas)
        ].copy()

        partes_mycom.append(df)

    mycom_raw = pd.concat(
        partes_mycom,
        ignore_index=True
    )

    # 4. Identificar jornadas completas
    control = (
        mycom_raw.groupby(claves)["hora_normalizada"]
        .agg(
            registros="size",
            horas_distintas="nunique",
            esquema=lambda s: set(s) == horarios_validos
        )
    )

    completas = control.loc[
        (control["registros"] == 25)
        & (control["horas_distintas"] == 25)
        & control["esquema"]
    ].reset_index()[claves]

    # 5. Recuperar cierres de las jornadas completas
    mycom_cierres = mycom_raw.merge(
        completas,
        on=claves,
        how="inner",
        validate="many_to_one"
    )

    mycom_cierres = mycom_cierres.loc[
        mycom_cierres["hora_normalizada"] == cierre,
        claves + columnas_mycom
    ].copy()

    # 6. Convertir las lecturas a valores numéricos
    for columna in columnas_mycom:
        mycom_cierres[columna] = pd.to_numeric(
            mycom_cierres[columna],
            errors="coerce"
        )

    # 7. Identificar el área de las variables
    mycom_cierres = mycom_cierres.rename(
        columns={
            c: f"Mycom | {c}"
            for c in columnas_mycom
        }
    )

    assert not mycom_cierres.duplicated(claves).any()

    # 8. Integrar sin perder fechas
    resultado = dataset_integrado.merge(
        mycom_cierres,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    columnas_agregadas = [
        f"Mycom | {c}" for c in columnas_mycom
    ]

    # 9. Mostrar resultados
    print("\nINTEGRACIÓN DE MYCOM")

    print("Fechas:", len(resultado))
    print("Variables añadidas:", len(columnas_agregadas))
    print("Columnas totales:", resultado.shape[1])

    print(
        "Pares D -> D+1:",
        resultado["target_manana"].notna().sum()
    )

    print(
        "Fechas sin cierre Mycom:",
        resultado[columnas_agregadas].isna().all(axis=1).sum()
    )

    print("\nFaltantes por compresor:")
    print(resultado[columnas_agregadas].isna().sum())

    # 10. Verificaciones contra el notebook
    assert len(resultado) == 1126
    assert len(columnas_agregadas) == 7
    assert resultado.shape[1] == 39
    assert resultado["fecha"].is_unique

    assert (
        resultado["target_manana"].notna().sum() == 1054
    )

    return resultado


def integrar_agua(dataset_integrado, dataset_base):
    """
    Integra los consumos diarios de agua respetando
    fecha, fuente y cierre de las 23:59.
    """

    claves = ["archivo_origen", "fecha"]
    cierre = pd.Timedelta(hours=23, minutes=59)
    un_segundo = pd.Timedelta(seconds=1)

    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }
    horarios_validos.add(cierre)

    partes_agua = []

    # 1. Leer la hoja de agua de cada Excel
    for archivo in ARCHIVOS:

        df = pd.read_excel(
            archivo,
            sheet_name="Consolidado Agua"
        )

        df.columns = df.columns.astype(str).str.strip()

        df["fecha"] = pd.to_datetime(
            df["DIA"], errors="coerce"
        ).dt.normalize()

        # 2. Filtrar fechas seleccionadas de esta fuente
        fechas_elegidas = set(
            dataset_base.loc[
                dataset_base["archivo_origen"] == archivo.name,
                "fecha"
            ]
        )

        df = df.loc[
            df["fecha"].isin(fechas_elegidas)
        ].copy()

        # 3. Normalizar horarios
        horas = pd.to_timedelta(
            df["HORA"].astype(str),
            errors="coerce"
        )

        desfase = (
            horas - un_segundo
        ).isin(horarios_validos)

        df["hora_normalizada"] = horas.where(
            ~desfase,
            horas - un_segundo
        )

        # 4. Identificar jornadas completas
        control = (
            df.groupby("fecha")["hora_normalizada"]
            .agg(
                registros="size",
                horas_distintas="nunique",
                esquema=lambda s: set(s) == horarios_validos
            )
        )

        fechas_completas = control.index[
            (control["registros"] == 25)
            & (control["horas_distintas"] == 25)
            & control["esquema"]
        ]

        # 5. Recuperar cierres de las 23:59
        df = df.loc[
            df["fecha"].isin(fechas_completas)
            & df["hora_normalizada"].eq(cierre)
        ].copy()

        # 6. Seleccionar variables del área
        excluir = {
            "DIA", "HORA", "Id",
            "fecha", "hora_normalizada"
        }

        columnas = [
            c for c in df.columns
            if c not in excluir
            and not c.startswith("Unnamed")
        ]

        df = df[["fecha"] + columnas].copy()
        df.insert(0, "archivo_origen", archivo.name)

        # 7. Convertir a valores numéricos
        for columna in columnas:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

        # 8. Identificar el área
        df = df.rename(
            columns={
                c: f"Agua | {c}"
                for c in columnas
            }
        )

        partes_agua.append(df)

    # 9. Unificar las tres fuentes
    # sort=False permite conservar columnas que
    # existen solo en algunas versiones del Excel
    agua = pd.concat(
        partes_agua,
        ignore_index=True,
        sort=False
    )

    assert not agua.duplicated(claves).any()

    # 10. Integrar sin eliminar días del dataset base
    resultado = dataset_integrado.merge(
        agua,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    columnas_agua = [
        c for c in resultado.columns
        if c.startswith("Agua | ")
    ]

    sin_datos = resultado[
        columnas_agua
    ].isna().all(axis=1).sum()

    # 11. Mostrar resultados
    print("\nINTEGRACIÓN DE AGUA")
    print("Fechas:", len(resultado))
    print("Variables añadidas:", len(columnas_agua))
    print("Columnas totales:", resultado.shape[1])

    print(
        "Pares D -> D+1:",
        resultado["target_manana"].notna().sum()
    )

    print("Fechas sin datos de agua:", sin_datos)

    print("\nVariables con más faltantes:")
    print(
        resultado[columnas_agua]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # 12. Comprobaciones contra el notebook
    assert len(resultado) == 1126
    assert len(columnas_agua) == 22
    assert resultado.shape[1] == 61
    assert sin_datos == 1
    assert resultado["fecha"].is_unique

    assert (
        resultado["target_manana"].notna().sum()
        == 1054
    )

    return resultado


def integrar_area_consolidada(
    dataset_integrado,
    dataset_base,
    hoja,
    prefijo
):
    """
    Integra una hoja industrial usando:
    - La fecha y fuente seleccionadas.
    - Jornadas con 25 horarios correctos.
    - El cierre del día D a las 23:59.
    """

    claves = ["archivo_origen", "fecha"]
    cierre = pd.Timedelta(hours=23, minutes=59)
    un_segundo = pd.Timedelta(seconds=1)

    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }
    horarios_validos.add(cierre)

    partes = []

    for archivo in ARCHIVOS:

        # 1. Leer la hoja
        df = pd.read_excel(
            archivo,
            sheet_name=hoja
        )

        df.columns = df.columns.astype(str).str.strip()

        df["fecha"] = pd.to_datetime(
            df["DIA"],
            errors="coerce"
        ).dt.normalize()

        # 2. Conservar fechas de la fuente seleccionada
        fechas_elegidas = set(
            dataset_base.loc[
                dataset_base["archivo_origen"] == archivo.name,
                "fecha"
            ]
        )

        df = df.loc[
            df["fecha"].isin(fechas_elegidas)
        ].copy()

        # 3. Normalizar horarios
        horas = pd.to_timedelta(
            df["HORA"].astype(str),
            errors="coerce"
        )

        desfase = (
            horas - un_segundo
        ).isin(horarios_validos)

        df["hora_normalizada"] = horas.where(
            ~desfase,
            horas - un_segundo
        )

        # 4. Controlar jornadas completas
        control = (
            df.groupby("fecha")["hora_normalizada"]
            .agg(
                registros="size",
                horas_distintas="nunique",
                esquema=lambda s: set(s) == horarios_validos
            )
        )

        fechas_completas = control.index[
            (control["registros"] == 25)
            & (control["horas_distintas"] == 25)
            & control["esquema"]
        ]

        # 5. Seleccionar cierres válidos
        df = df.loc[
            df["fecha"].isin(fechas_completas)
            & df["hora_normalizada"].eq(cierre)
        ].copy()

        # 6. Identificar las variables industriales
        excluir = {
            "DIA", "HORA", "Id",
            "fecha", "hora_normalizada"
        }

        columnas = [
            c for c in df.columns
            if c not in excluir
            and not c.startswith("Unnamed")
        ]

        df = df[["fecha"] + columnas].copy()
        df.insert(0, "archivo_origen", archivo.name)

        # 7. Convertir a números
        for columna in columnas:
            df[columna] = pd.to_numeric(
                df[columna],
                errors="coerce"
            )

        # 8. Agregar prefijo del área
        df = df.rename(
            columns={
                c: f"{prefijo} | {c}"
                for c in columnas
            }
        )

        partes.append(df)

    # 9. Unificar fuentes
    datos_area = pd.concat(
        partes,
        ignore_index=True,
        sort=False
    )

    assert not datos_area.duplicated(claves).any()

    # 10. Integrar al dataset sin perder observaciones
    resultado = dataset_integrado.merge(
        datos_area,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    columnas_area = [
        c for c in resultado.columns
        if c.startswith(f"{prefijo} | ")
    ]

    sin_datos = (
        resultado[columnas_area]
        .isna()
        .all(axis=1)
        .sum()
    )

    print(f"\nINTEGRACIÓN DE {prefijo.upper()}")
    print("Fechas:", len(resultado))
    print("Variables añadidas:", len(columnas_area))
    print("Columnas totales:", resultado.shape[1])
    print("Fechas sin datos del área:", sin_datos)

    print(
        "Pares D -> D+1:",
        resultado["target_manana"].notna().sum()
    )

    print("\nVariables con más faltantes:")
    print(
        resultado[columnas_area]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # Verificaciones generales
    assert len(resultado) == len(dataset_base)
    assert resultado["fecha"].is_unique
    assert (
        resultado["target_manana"].notna().sum()
        == 1054
    )

    return resultado




def dividir_temporal(dataset_integrado):
    """
    Construye Train, Validation y Test respetando
    el orden temporal y evitando solapamientos
    entre fechas objetivo y particiones futuras.
    """

    # 1. Conservar únicamente pares D -> D+1
    datos = (
        dataset_integrado
        .dropna(subset=["target_manana"])
        .sort_values("fecha")
        .reset_index(drop=True)
        .copy()
    )

    n = len(datos)

    # 2. Cortes cronológicos iniciales: 70/15/15
    corte_train = int(n * 0.70)
    corte_val = int(n * 0.85)

    train = datos.iloc[:corte_train].copy()
    val = datos.iloc[corte_train:corte_val].copy()
    test = datos.iloc[corte_val:].copy()

    # 3. Evitar solapamientos en las fronteras
    # El objetivo de cada fila corresponde a
    # la fecha_siguiente.

    descartadas_train = 0
    while (
        not train.empty
        and train["fecha_siguiente"].iloc[-1]
        >= val["fecha"].iloc[0]
    ):
        train = train.iloc[:-1].copy()
        descartadas_train += 1

    descartadas_val = 0
    while (
        not val.empty
        and val["fecha_siguiente"].iloc[-1]
        >= test["fecha"].iloc[0]
    ):
        val = val.iloc[:-1].copy()
        descartadas_val += 1

    # 4. Identificar predictores y metadatos
    columnas_excluir = [
        "fecha",
        "archivo_origen",
        "fecha_siguiente",
        "target_manana"
    ]

    features = [
        c for c in datos.columns
        if c not in columnas_excluir
    ]

    # 5. Mostrar resultados
    print("\nDIVISIÓN TEMPORAL")
    print("Pares supervisados originales:", n)
    print(
        "Observaciones excluidas en fronteras:",
        descartadas_train + descartadas_val
    )

    for nombre, parte in [
        ("Train", train),
        ("Validation", val),
        ("Test", test)
    ]:
        print(
            f"{nombre}: {len(parte)} filas |",
            parte["fecha"].min().date(),
            "a",
            parte["fecha"].max().date()
        )

    print("\nPredictores industriales:", len(features))
    print(
        "Total de observaciones finales:",
        len(train) + len(val) + len(test)
    )

    # 6. Verificaciones de reproducibilidad
    assert n == 1054

    assert (
        len(train), len(val), len(test)
    ) == (736, 157, 159)

    assert len(features) == 88

    assert (
        train["fecha_siguiente"].max()
        < val["fecha"].min()
    )

    assert (
        val["fecha_siguiente"].max()
        < test["fecha"].min()
    )

    return train, val, test, features


def filtrar_variables_faltantes(train, val, features):
    """Aprende el filtro de faltantes en Train, y lo aplica a Validation.

    Test se mantiene reservado hasta terminar la selección del modelo.
    """
    porcentaje_faltantes = train[features].isna().mean()
    umbral = 0.50
    columnas_descartadas = (
        porcentaje_faltantes[porcentaje_faltantes > umbral]
        .sort_values(ascending=False)
    )
    features_filtradas = [
        c for c in features if porcentaje_faltantes[c] <= umbral
    ]
    X_train = train[features_filtradas].copy()
    X_val = val[features_filtradas].copy()

    print("\nFILTRO DE VARIABLES CON FALTANTES")
    print("Variables originales:", len(features))
    print("Umbral máximo de faltantes:", umbral)
    print("\nVariables descartadas:")
    for columna, porcentaje in columnas_descartadas.items():
        print(f"- {columna}: {porcentaje:.2%}")
    print("Variables conservadas:", len(features_filtradas))
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)

    assert len(features) == 88
    assert len(columnas_descartadas) == 1
    assert len(features_filtradas) == 87
    assert "GasVapor | Vapor _Vapor_L5 (KG)" in columnas_descartadas.index
    assert X_train.shape == (736, 87)
    assert X_val.shape == (157, 87)
    assert X_train.columns.equals(X_val.columns)
    return X_train, X_val, features_filtradas, columnas_descartadas

def tratar_valores_extremos(X_train, X_val):
    """Calcula 10×P99 solo en Train y aplica el mismo límite a Validation."""
    percentiles_99 = X_train.quantile(0.99)
    umbrales_extremos = (10 * percentiles_99).where(percentiles_99 > 0)

    def aplicar_umbrales(X):
        mascara = X.gt(umbrales_extremos, axis="columns")
        return X.mask(mascara), mascara

    X_train_limpio, extremos_train = aplicar_umbrales(X_train)
    X_val_limpio, extremos_val = aplicar_umbrales(X_val)

    resumen_extremos = pd.DataFrame({
        "percentil_99_train": percentiles_99,
        "umbral_superior": umbrales_extremos,
        "extremos_train": extremos_train.sum(),
        "extremos_val": extremos_val.sum()
    })
    resumen_extremos = resumen_extremos.loc[
        resumen_extremos["extremos_train"] > 0
    ].sort_values("extremos_train", ascending=False)

    print("\nTRATAMIENTO DE VALORES EXTREMOS")
    print("Extremos detectados en Train:", int(extremos_train.sum().sum()))
    print("Extremos detectados en Validation:", int(extremos_val.sum().sum()))
    print("Variables afectadas en Train:", len(resumen_extremos))
    print(resumen_extremos.to_string())

    assert X_train_limpio.shape == X_train.shape
    assert X_val_limpio.shape == X_val.shape
    assert X_train_limpio.columns.equals(X_val_limpio.columns)
    return X_train_limpio, X_val_limpio, umbrales_extremos, resumen_extremos

def imputar_faltantes(X_train_limpio, X_val_limpio):
    """Imputa Train/Validation con medianas Train y crea indicadores Train."""
    medianas_train = X_train_limpio.median()
    assert medianas_train.notna().all(), (
        "Hay variables sin valores válidos en Train."
    )
    columnas_indicadores = (
        X_train_limpio.columns[X_train_limpio.isna().any()].tolist()
    )

    def transformar(X):
        resultado = X.fillna(medianas_train).copy()
        indicadores = X[columnas_indicadores].isna().astype(int)
        indicadores.columns = [f"{c} | faltante" for c in columnas_indicadores]
        return pd.concat([resultado, indicadores], axis=1)

    X_train_final = transformar(X_train_limpio)
    X_val_final = transformar(X_val_limpio)

    print("\nIMPUTACIÓN DE VALORES FALTANTES")
    print("Variables originales:", X_train_limpio.shape[1])
    print("Indicadores añadidos:", len(columnas_indicadores))
    print("Train:", X_train_final.shape)
    print("Validation:", X_val_final.shape)
    print("Faltantes Train:", int(X_train_final.isna().sum().sum()))
    print("Faltantes Validation:", int(X_val_final.isna().sum().sum()))

    assert X_train_final.shape == (736, 131)
    assert X_val_final.shape == (157, 131)
    assert len(columnas_indicadores) == 44
    assert X_train_final.columns.equals(X_val_final.columns)
    assert not X_train_final.isna().any().any()
    assert not X_val_final.isna().any().any()
    return X_train_final, X_val_final, medianas_train, columnas_indicadores

def agregar_calendario(X, fechas):
    """
    Incorpora cinco características temporales:
    mes, día de semana, fin de semana y
    codificación cíclica anual.
    """

    resultado = X.copy()
    fechas = pd.to_datetime(fechas)

    # 1. Variables de calendario
    resultado["mes"] = fechas.dt.month.to_numpy()

    resultado["dia_semana"] = (
        fechas.dt.day_of_week.to_numpy()
    )

    resultado["fin_semana"] = (
        fechas.dt.day_of_week >= 5
    ).astype(int).to_numpy()

    # 2. Considerar años bisiestos
    dias_del_anio = np.where(
        fechas.dt.is_leap_year,
        366,
        365
    )

    # 3. Posición cíclica dentro del año
    angulo = (
        2 * np.pi
        * (fechas.dt.day_of_year.to_numpy() - 1)
        / dias_del_anio
    )

    resultado["sen_anual"] = np.sin(angulo)
    resultado["cos_anual"] = np.cos(angulo)

    return resultado


def agregar_historicas(X_train_cal, X_val_cal, train, val, frio_diario):
    """Construye lags estrictamente pasados para Train y Validation.

    La serie de consumo se desplaza un día antes de aplicar medias móviles.
    Test no se transforma hasta que finaliza la selección.
    """
    serie_frio = (
        frio_diario.set_index("fecha")["frio_kw"].sort_index().asfreq("D")
    )
    historicas = pd.DataFrame(index=serie_frio.index)
    historicas["frio_lag_1d"] = serie_frio.shift(1)
    historicas["frio_lag_7d"] = serie_frio.shift(7)
    historicas["frio_media_3d"] = (
        serie_frio.shift(1).rolling(3, min_periods=3).mean()
    )
    historicas["frio_media_7d"] = (
        serie_frio.shift(1).rolling(7, min_periods=7).mean()
    )
    historicas = historicas.reset_index()
    columnas_historicas = [
        "frio_lag_1d", "frio_lag_7d", "frio_media_3d", "frio_media_7d"
    ]

    def incorporar(X, particion):
        valores = particion[["fecha"]].merge(
            historicas, on="fecha", how="left", validate="one_to_one"
        )
        resultado = X.copy()
        for columna in columnas_historicas:
            resultado[columna] = valores[columna].to_numpy()
        return resultado

    X_train_hist = incorporar(X_train_cal, train)
    X_val_hist = incorporar(X_val_cal, val)
    print("\nVARIABLES HISTÓRICAS")
    print("Características añadidas: 4")
    print("Train:", X_train_hist.shape)
    print("Validation:", X_val_hist.shape)
    print("\nFALTANTES HISTÓRICOS")
    print(pd.DataFrame({
        "Train": X_train_hist[columnas_historicas].isna().sum(),
        "Validation": X_val_hist[columnas_historicas].isna().sum()
    }).to_string())
    assert X_train_hist.shape == (736, 140)
    assert X_val_hist.shape == (157, 140)
    assert X_train_hist.columns.equals(X_val_hist.columns)
    return X_train_hist, X_val_hist

def imputar_historicas(X_train_hist, X_val_hist):
    """Medianas históricas ajustadas en Train y aplicadas a Validation."""
    columnas_historicas = [
        "frio_lag_1d", "frio_lag_7d", "frio_media_3d", "frio_media_7d"
    ]
    medianas_historicas = X_train_hist[columnas_historicas].median()
    assert medianas_historicas.notna().all()

    def transformar(X):
        resultado = X.copy()
        for columna in columnas_historicas:
            resultado[f"{columna} | faltante"] = X[columna].isna().astype(int)
            resultado[columna] = X[columna].fillna(medianas_historicas[columna])
        return resultado

    X_train_preparado = transformar(X_train_hist)
    X_val_preparado = transformar(X_val_hist)
    print("\nIMPUTACIÓN DE VARIABLES HISTÓRICAS")
    print("Medianas de Train:")
    print(medianas_historicas.round(4).to_string())
    print("Train:", X_train_preparado.shape)
    print("Validation:", X_val_preparado.shape)
    print("Faltantes Train:", int(X_train_preparado.isna().sum().sum()))
    print("Faltantes Validation:", int(X_val_preparado.isna().sum().sum()))
    assert X_train_preparado.shape == (736, 144)
    assert X_val_preparado.shape == (157, 144)
    assert X_train_preparado.columns.equals(X_val_preparado.columns)
    assert not X_train_preparado.isna().any().any()
    assert not X_val_preparado.isna().any().any()
    return X_train_preparado, X_val_preparado, medianas_historicas

def integrar_kpi(dataset_integrado, dataset_base):
    """
    Integra cuatro indicadores KPI y sus metas,
    respetando fecha, fuente y cierre de las 23:59.

    No inventa metas ausentes en los Excel antiguos.
    """

    claves = ["archivo_origen", "fecha"]

    pares_kpi = {
        "EE Planta / Hl": "Meta EE Planta",
        "Agua Planta / Hl": "Meta Agua Planta",
        "ET Planta / Hl": "Meta ET Planta",
        "Aire Planta / Hl": "Meta Aire Planta"
    }

    columnas_kpi = (
        list(pares_kpi.keys())
        + list(pares_kpi.values())
    )

    cierre = pd.Timedelta(hours=23, minutes=59)
    un_segundo = pd.Timedelta(seconds=1)

    horarios_validos = {
        pd.Timedelta(hours=h) for h in range(24)
    }
    horarios_validos.add(cierre)

    partes_kpi = []

    # 1. Leer cada Excel
    for archivo in ARCHIVOS:

        print(f"Leyendo KPI: {archivo.name}")

        df = pd.read_excel(
            archivo,
            sheet_name="Consolidado KPI"
        )

        df.columns = (
            df.columns.astype(str).str.strip()
        )

        # Los indicadores reales deben existir
        faltantes = [
            c for c in pares_kpi
            if c not in df.columns
        ]

        if faltantes:
            raise ValueError(
                f"Faltan indicadores en {archivo.name}: "
                f"{faltantes}"
            )

        # 2. Normalizar fecha y horarios
        fechas = pd.to_datetime(
            df["DIA"], errors="coerce"
        ).dt.normalize()

        horas = pd.to_timedelta(
            df["HORA"].astype(str),
            errors="coerce"
        )

        desfase = (
            horas - un_segundo
        ).isin(horarios_validos)

        horas = horas.where(
            ~desfase,
            horas - un_segundo
        )

        # 3. Crear tabla con las variables necesarias
        datos = pd.DataFrame({
            "fecha": fechas,
            "hora_normalizada": horas
        })

        for columna in columnas_kpi:
            if columna in df.columns:
                datos[columna] = pd.to_numeric(
                    df[columna],
                    errors="coerce"
                )
            else:
                datos[columna] = np.nan

        # 4. Usar solo fechas elegidas de este archivo
        fechas_elegidas = set(
            dataset_base.loc[
                dataset_base["archivo_origen"] == archivo.name,
                "fecha"
            ]
        )

        datos = datos.loc[
            datos["fecha"].isin(fechas_elegidas)
        ].copy()

        # 5. Exigir las 25 lecturas esperadas
        control = (
            datos.groupby("fecha")["hora_normalizada"]
            .agg(
                registros="size",
                horas="nunique",
                esquema=lambda s: set(s) == horarios_validos
            )
        )

        fechas_completas = control.index[
            (control["registros"] == 25)
            & (control["horas"] == 25)
            & control["esquema"]
        ]

        # 6. Recuperar el cierre a las 23:59
        datos = datos.loc[
            datos["fecha"].isin(fechas_completas)
            & datos["hora_normalizada"].eq(cierre)
        ].copy()

        datos["archivo_origen"] = archivo.name

        datos = datos[
            claves + columnas_kpi
        ].copy()

        # 7. Identificar columnas del área
        datos = datos.rename(
            columns={
                c: f"KPI | {c}"
                for c in columnas_kpi
            }
        )

        partes_kpi.append(datos)

    # 8. Unificar fuentes
    kpi_diario = pd.concat(
        partes_kpi,
        ignore_index=True
    )

    assert not kpi_diario.duplicated(claves).any()

    # 9. Integrar sin perder fechas
    dataset_kpi = dataset_integrado.merge(
        kpi_diario,
        on=claves,
        how="left",
        validate="one_to_one"
    )

    indicadores_reales = [
        f"KPI | {c}" for c in pares_kpi
    ]

    indicadores_meta = [
        f"KPI | {c}" for c in pares_kpi.values()
    ]

    # 10. Mostrar resultados
    print("\nINTEGRACIÓN DE KPI")

    print("Fechas:", len(dataset_kpi))
    print("Variables añadidas:", len(columnas_kpi))
    print("Columnas totales:", dataset_kpi.shape[1])

    print(
        "Fechas con los 4 indicadores reales:",
        dataset_kpi[indicadores_reales]
        .notna().all(axis=1).sum()
    )

    print(
        "Fechas con las 4 metas:",
        dataset_kpi[indicadores_meta]
        .notna().all(axis=1).sum()
    )

    print("\nFALTANTES POR INDICADOR")
    print(
        dataset_kpi[
            indicadores_reales + indicadores_meta
        ].isna().sum().to_string()
    )

    # 11. Validaciones contra el notebook
    assert dataset_kpi.shape == (1126, 100)
    assert dataset_kpi["fecha"].is_unique

    assert (
        dataset_kpi[indicadores_reales]
        .notna().all(axis=1).sum() == 1098
    )

    assert (
        dataset_kpi[indicadores_meta]
        .notna().all(axis=1).sum() == 789
    )

    return dataset_kpi


def crear_variables_eficiencia(dataset_kpi):
    """
    Calcula la desviación porcentual de cuatro
    indicadores respecto de sus metas.

    No calcula desviaciones con metas nulas,
    ausentes o no positivas.
    """

    pares_eficiencia = {
        "EE Planta": (
            "EE Planta / Hl",
            "Meta EE Planta"
        ),
        "Agua Planta": (
            "Agua Planta / Hl",
            "Meta Agua Planta"
        ),
        "ET Planta": (
            "ET Planta / Hl",
            "Meta ET Planta"
        ),
        "Aire Planta": (
            "Aire Planta / Hl",
            "Meta Aire Planta"
        )
    }

    dataset_eficiencia = dataset_kpi.copy()

    columnas_eficiencia = []

    # 1. Calcular las cuatro desviaciones
    for nombre, (indicador, meta) in pares_eficiencia.items():

        real = dataset_eficiencia[
            f"KPI | {indicador}"
        ]

        objetivo = dataset_eficiencia[
            f"KPI | {meta}"
        ]

        columna = (
            f"Eficiencia | Desviacion {nombre} (%)"
        )

        # Evitar división por metas no positivas
        meta_valida = objetivo.where(objetivo > 0)

        desviacion = (
            (real - meta_valida)
            / meta_valida
            * 100
        )

        # Evitar valores infinitos
        dataset_eficiencia[columna] = (
            desviacion.replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

        columnas_eficiencia.append(columna)

    # 2. Resumen de disponibilidad
    resumen = pd.DataFrame({
        "disponibles": (
            dataset_eficiencia[columnas_eficiencia]
            .notna().sum()
        ),
        "faltantes": (
            dataset_eficiencia[columnas_eficiencia]
            .isna().sum()
        )
    })

    # 3. Mostrar resultados
    print("\nVARIABLES DE EFICIENCIA")
    print("Variables añadidas:", len(columnas_eficiencia))
    print("Fechas:", len(dataset_eficiencia))
    print("Columnas totales:", dataset_eficiencia.shape[1])

    print("\nDISPONIBILIDAD")
    print(resumen.to_string())

    print("\nPRIMEROS EJEMPLOS")

    ejemplos = dataset_eficiencia.loc[
        dataset_eficiencia[
            columnas_eficiencia
        ].notna().all(axis=1),
        ["fecha"] + columnas_eficiencia
    ].head()

    print(ejemplos.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 4. Verificaciones contra el notebook
    assert dataset_eficiencia.shape == (1126, 104)
    assert len(columnas_eficiencia) == 4
    assert dataset_eficiencia["fecha"].is_unique

    assert (
        dataset_eficiencia[columnas_eficiencia]
        .notna().sum().eq(764).all()
    )

    return dataset_eficiencia, columnas_eficiencia


def calcular_ratios_industriales(X):
    """
    Construye cuatro ratios industriales a partir
    de variables base previamente depuradas.

    Si falta algún componente, el ratio queda en NaN.
    No permite denominadores nulos o negativos.
    """

    formulas = {
        "Ratio | Frio sobre Servicios": (
            "frio_kw",
            "EE | Servicios (Kw)"
        ),
        "Ratio | Aire sobre Servicios": (
            "EE | Aire (Kw)",
            "EE | Servicios (Kw)"
        ),
        "Ratio | Envasado sobre Planta": (
            "EE | Envasado (Kw)",
            "EE | Planta (Kw)"
        ),
        "Ratio | Frio por Hl Bodega": (
            "frio_kw",
            "Produccion | Hl Producido Bodega"
        )
    }

    ratios = pd.DataFrame(index=X.index)

    # 1. Calcular cada ratio
    for nombre, (numerador_col, denominador_col) in formulas.items():

        numerador = pd.to_numeric(
            X[numerador_col],
            errors="coerce"
        )

        denominador = pd.to_numeric(
            X[denominador_col],
            errors="coerce"
        )

        # No admitir consumos negativos
        numerador = numerador.where(numerador >= 0)

        # El denominador debe ser positivo
        denominador = denominador.where(denominador > 0)

        ratio = numerador / denominador

        # Evitar infinitos
        ratios[nombre] = ratio.replace(
            [np.inf, -np.inf],
            np.nan
        )

    return ratios


def calcular_interacciones_industriales(X):
    """
    Construye tres interacciones industriales
    utilizando variables base depuradas y
    todavía no imputadas.

    Si alguno de los componentes es inválido,
    la interacción queda como NaN.
    """

    pares_interacciones = {
        "Interaccion | Frio x Mycom 7": (
            "frio_kw",
            "Mycom | KW Mycom 7"
        ),
        "Interaccion | Frio x Produccion Bodega": (
            "frio_kw",
            "Produccion | Hl Producido Bodega"
        ),
        "Interaccion | Aire x Cerveza Envasada": (
            "EE | Aire (Kw)",
            "Produccion | Hl Cerveza Envasada"
        )
    }

    interacciones = pd.DataFrame(index=X.index)

    # 1. Calcular las tres interacciones
    for nombre, (col_a, col_b) in pares_interacciones.items():

        a = pd.to_numeric(
            X[col_a], errors="coerce"
        )

        b = pd.to_numeric(
            X[col_b], errors="coerce"
        )

        # Los consumos y volúmenes no deben
        # ser negativos
        a = a.where(a >= 0)
        b = b.where(b >= 0)

        # Si falta alguno de los valores,
        # el producto permanece como NaN
        producto = a * b

        interacciones[nombre] = producto.replace(
            [np.inf, -np.inf],
            np.nan
        )

    return interacciones


def unir_caracteristicas_derivadas(
    X,
    particion,
    dataset_eficiencia,
    columnas_eficiencia,
    ratios,
    interacciones
):
    """
    Incorpora:
    - 4 características de eficiencia KPI
    - 4 ratios industriales
    - 3 interacciones industriales

    Conserva el orden de las observaciones
    y mantiene los valores faltantes.
    """

    claves = ["fecha", "archivo_origen"]

    # 1. Comprobar alineación de los datos
    assert X.index.equals(particion.index)
    assert X.index.equals(ratios.index)
    assert X.index.equals(interacciones.index)

    # 2. Recuperar KPI por fecha y archivo
    kpi_particion = particion[claves].merge(
        dataset_eficiencia[
            claves + columnas_eficiencia
        ],
        on=claves,
        how="left",
        validate="one_to_one",
        sort=False
    )

    assert len(kpi_particion) == len(X)

    # 3. Recuperar el índice original de X
    eficiencia = pd.DataFrame(
        kpi_particion[columnas_eficiencia].to_numpy(),
        index=X.index,
        columns=columnas_eficiencia
    )

    # 4. Unir las 11 características
    resultado = pd.concat(
        [
            X,
            eficiencia,
            ratios,
            interacciones
        ],
        axis=1
    )

    columnas_nuevas = (
        columnas_eficiencia
        + ratios.columns.tolist()
        + interacciones.columns.tolist()
    )

    # 5. Verificar consistencia
    assert len(columnas_nuevas) == 11
    assert resultado.shape == (len(X), 155)
    assert resultado.columns.is_unique
    assert resultado.index.equals(X.index)

    return resultado, columnas_nuevas


def preparar_caracteristicas_derivadas(
    X_train_ampliado,
    X_val_ampliado,
    nuevas_features
):
    """
    Limpia, detecta extremos e imputa las
    11 características derivadas.

    Todos los parámetros se calculan con Train.
    """

    train_limpio = X_train_ampliado.copy()
    val_limpio = X_val_ampliado.copy()

    columnas_eficiencia = [
        c for c in nuevas_features
        if c.startswith("Eficiencia | ")
    ]

    columnas_ratios_interacciones = [
        c for c in nuevas_features
        if c.startswith(("Ratio | ", "Interaccion | "))
    ]

    # 1. Limpiar valores imposibles
    def limpiar_valores(X):
        resultado = X.copy()

        resultado[nuevas_features] = (
            resultado[nuevas_features]
            .replace([np.inf, -np.inf], np.nan)
        )

        # Desviaciones inferiores a -100 %
        for columna in columnas_eficiencia:
            resultado[columna] = resultado[columna].mask(
                resultado[columna] < -100
            )

        # Ratios e interacciones negativos
        for columna in columnas_ratios_interacciones:
            resultado[columna] = resultado[columna].mask(
                resultado[columna] < 0
            )

        return resultado

    train_limpio = limpiar_valores(train_limpio)
    val_limpio = limpiar_valores(val_limpio)

    # 2. Calcular umbrales exclusivamente con Train
    p99_train = train_limpio[nuevas_features].quantile(0.99)

    umbrales_derivadas = (10 * p99_train).where(
        p99_train > 0
    )

    # 3. Detectar y tratar extremos
    mascara_train = train_limpio[nuevas_features].gt(
        umbrales_derivadas,
        axis="columns"
    )

    mascara_val = val_limpio[nuevas_features].gt(
        umbrales_derivadas,
        axis="columns"
    )

    train_limpio[nuevas_features] = (
        train_limpio[nuevas_features].mask(mascara_train)
    )

    val_limpio[nuevas_features] = (
        val_limpio[nuevas_features].mask(mascara_val)
    )

    # 4. Identificar variables con faltantes en Train
    columnas_indicadores = [
        c for c in nuevas_features
        if train_limpio[c].isna().any()
    ]

    # 5. Calcular medianas exclusivamente con Train
    medianas_derivadas = (
        train_limpio[nuevas_features].median()
    )

    if medianas_derivadas.isna().any():
        raise ValueError(
            "Hay características derivadas sin "
            "valores válidos en Train."
        )

    # 6. Imputar y agregar indicadores
    def transformar(X):
        resultado = X.copy()

        for columna in columnas_indicadores:
            resultado[f"{columna} | faltante"] = (
                resultado[columna].isna().astype(int)
            )

        resultado[nuevas_features] = (
            resultado[nuevas_features]
            .fillna(medianas_derivadas)
        )

        return resultado

    X_train_completo = transformar(train_limpio)
    X_val_completo = transformar(val_limpio)

    # 7. Mostrar resultados
    print("\nPREPARACIÓN DE CARACTERÍSTICAS DERIVADAS")

    print("Extremos en Train:", int(mascara_train.sum().sum()))
    print("Extremos en Validation:", int(mascara_val.sum().sum()))

    print("\nIndicadores añadidos:", len(columnas_indicadores))
    print("Variables con indicador:")
    for columna in columnas_indicadores:
        print("-", columna)

    print("\nDIMENSIONES")
    print("Train:", X_train_completo.shape)
    print("Validation:", X_val_completo.shape)

    print("\nFALTANTES RESTANTES")
    print("Train:", X_train_completo.isna().sum().sum())
    print("Validation:", X_val_completo.isna().sum().sum())

    # 8. Comprobaciones
    assert X_train_completo.shape[0] == 736
    assert X_val_completo.shape[0] == 157

    assert X_train_completo.shape[1] == (
        155 + len(columnas_indicadores)
    )

    assert X_train_completo.columns.equals(
        X_val_completo.columns
    )

    assert not X_train_completo.isna().any().any()
    assert not X_val_completo.isna().any().any()

    return (
        X_train_completo,
        X_val_completo,
        medianas_derivadas,
        umbrales_derivadas,
        columnas_indicadores
    )


from sklearn.ensemble import RandomForestRegressor


def seleccionar_variables_rf(X_train, y_train):
    """
    Selecciona las 22 variables más importantes
    mediante Random Forest.

    El modelo se ajusta únicamente con Train.
    """

    # 1. Entrenar Random Forest
    selector_rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    selector_rf.fit(X_train, y_train)

    # 2. Calcular importancia de características
    importancias_rf = pd.Series(
        selector_rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    # 3. Seleccionar las 22 más importantes
    features_rf = (
        importancias_rf.head(22).index.tolist()
    )

    # 4. Mostrar resultados
    print("\nSELECCIÓN DE VARIABLES - RANDOM FOREST")

    print("Variables evaluadas:", X_train.shape[1])
    print("Variables seleccionadas:", len(features_rf))

    print("\nTOP 22 - RANDOM FOREST")

    ranking_rf = pd.DataFrame({
        "variable": features_rf,
        "importancia_rf": importancias_rf.loc[
            features_rf
        ].values
    })

    print(
        ranking_rf.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    # 5. Verificaciones
    assert len(features_rf) == 22
    assert len(set(features_rf)) == 22

    assert np.isclose(
        importancias_rf.sum(), 1.0
    )

    return features_rf, importancias_rf, selector_rf



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


def seleccionar_variables_lasso(
    X_train, y_train, X_val, y_val
):
    """
    Aplica LASSO con escalado dentro de Pipeline.
    Entrena con Train y selecciona alpha con Validation.
    """

    alphas = [1, 10, 30, 100, 300, 1000, 3000]

    resultados = []
    modelos = []

    for alpha in alphas:

        modelo = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", Lasso(
                alpha=alpha,
                max_iter=30000
            ))
        ])

        modelo.fit(X_train, y_train)

        pred = modelo.predict(X_val)

        rmse = np.sqrt(
            mean_squared_error(y_val, pred)
        )

        coeficientes = modelo.named_steps[
            "lasso"
        ].coef_

        seleccionadas = int(
            np.sum(np.abs(coeficientes) > 1e-6)
        )

        resultados.append({
            "alpha": alpha,
            "RMSE_validacion": rmse,
            "variables_seleccionadas": seleccionadas
        })

        modelos.append(modelo)

    # Comparación de regularizaciones
    tabla_lasso = pd.DataFrame(resultados)

    mejor_indice = tabla_lasso[
        "RMSE_validacion"
    ].idxmin()

    mejor_modelo = modelos[mejor_indice]

    mejor_alpha = tabla_lasso.loc[
        mejor_indice, "alpha"
    ]

    # Coeficientes del modelo seleccionado
    coeficientes_lasso = pd.Series(
        mejor_modelo.named_steps["lasso"].coef_,
        index=X_train.columns
    )

    seleccion_lasso = coeficientes_lasso.loc[
        coeficientes_lasso.abs() > 1e-6
    ].sort_values(
        key=lambda s: s.abs(),
        ascending=False
    )

    features_lasso = seleccion_lasso.index.tolist()

    print("\nSELECCIÓN DE VARIABLES - LASSO")

    print("\nCOMPARACIÓN DE ALPHAS")
    print(
        tabla_lasso.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nMejor alpha:", mejor_alpha)
    print(
        "Variables seleccionadas:",
        len(features_lasso)
    )

    print("\nTOP 20 - LASSO")
    print(
        seleccion_lasso.head(20).to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    return (
        features_lasso,
        coeficientes_lasso,
        mejor_modelo,
        tabla_lasso
    )



def comparar_selecciones(
    X_train,
    y_train,
    X_val,
    y_val,
    features_rf,
    features_lasso,
    modelo_lasso
):
    """
    Compara subconjuntos con un Random Forest
    de referencia y agrega LASSO como modelo
    independiente.

    Toda la evaluación se realiza en Validation.
    """

    seleccion_rf = set(features_rf)
    seleccion_lasso = set(features_lasso)

    coincidentes = seleccion_rf & seleccion_lasso
    union = seleccion_rf | seleccion_lasso

    print("\nCOMPARACIÓN DE SELECCIONES")
    print("Variables RF:", len(seleccion_rf))
    print("Variables LASSO:", len(seleccion_lasso))
    print("Coincidentes:", len(coincidentes))
    print("Unión:", len(union))

    print("\nVARIABLES COINCIDENTES")
    for variable in sorted(coincidentes):
        print("-", variable)

    conjuntos = {
        "Solo frio actual": ["frio_kw"],
        "Coincidentes": sorted(coincidentes),
        "Random Forest": features_rf,
        "LASSO (variables)": features_lasso,
        "Union": sorted(union),
        "Todas": X_train.columns.tolist()
    }

    resultados = []

    # Referencia: consumo de mañana = consumo de hoy
    pred_persistencia = X_val["frio_kw"].to_numpy()

    resultados.append({
        "conjunto": "Persistencia",
        "modelo": "Regla simple",
        "variables": 1,
        "RMSE": np.sqrt(
            mean_squared_error(
                y_val, pred_persistencia
            )
        ),
        "MAE": mean_absolute_error(
            y_val, pred_persistencia
        )
    })

    # Mismo Random Forest para cada subconjunto
    for nombre, columnas in conjuntos.items():

        if not columnas:
            continue

        modelo = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        modelo.fit(
            X_train[columnas],
            y_train
        )

        pred = modelo.predict(X_val[columnas])

        resultados.append({
            "conjunto": nombre,
            "modelo": "Random Forest",
            "variables": len(columnas),
            "RMSE": np.sqrt(
                mean_squared_error(y_val, pred)
            ),
            "MAE": mean_absolute_error(y_val, pred)
        })

    # Evaluar LASSO con su propio modelo
    pred_lasso = modelo_lasso.predict(X_val)

    resultados.append({
        "conjunto": "LASSO (modelo propio)",
        "modelo": "LASSO",
        "variables": len(features_lasso),
        "RMSE": np.sqrt(
            mean_squared_error(y_val, pred_lasso)
        ),
        "MAE": mean_absolute_error(y_val, pred_lasso)
    })

    tabla = (
        pd.DataFrame(resultados)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\nRESULTADOS EN VALIDACIÓN")
    print(
        tabla.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )

    return tabla, coincidentes, union


from sklearn.base import clone

def verificar_seleccion_final(
    X_train,
    y_train,
    X_val,
    y_val,
    features_lasso,
    mejor_lasso
):
    """
    Comprueba si LASSO conserva su rendimiento
    al entrenarse únicamente con las variables
    que seleccionó.

    No utiliza Test.
    """

    # 1. Selección de las 22 características
    features_finales = features_lasso.copy()

    X_train_sel = X_train[features_finales].copy()
    X_val_sel = X_val[features_finales].copy()

    # 2. Clonar el pipeline LASSO seleccionado
    # Conserva StandardScaler y alpha=300
    modelo_reducido = clone(mejor_lasso)

    modelo_reducido.fit(X_train_sel, y_train)

    # 3. Comparar predicciones en Validation
    pred_original = mejor_lasso.predict(X_val)
    pred_reducida = modelo_reducido.predict(X_val_sel)

    rmse_original = np.sqrt(
        mean_squared_error(y_val, pred_original)
    )

    rmse_reducido = np.sqrt(
        mean_squared_error(y_val, pred_reducida)
    )

    mae_reducido = mean_absolute_error(
        y_val, pred_reducida
    )

    # 4. Mostrar comparación
    print("\nVERIFICACIÓN DE SELECCIÓN FINAL")

    print("Variables originales:", X_train.shape[1])
    print("Variables seleccionadas:", len(features_finales))

    print("\nDIMENSIONES")
    print("Train:", X_train_sel.shape)
    print("Validation:", X_val_sel.shape)

    print("\nRENDIMIENTO")
    print(f"LASSO original - RMSE: {rmse_original:.2f}")
    print(f"LASSO reducido - RMSE: {rmse_reducido:.2f}")
    print(f"LASSO reducido - MAE: {mae_reducido:.2f}")

    print(
        "Diferencia absoluta de RMSE:",
        round(abs(rmse_reducido - rmse_original), 4)
    )

    print("\nVARIABLES FINALES")
    for i, variable in enumerate(features_finales, 1):
        print(f"{i:02d}. {variable}")

    # 5. Comprobaciones
    assert X_train_sel.shape == (736, 22)
    assert X_val_sel.shape == (157, 22)

    assert X_train_sel.columns.equals(
        X_val_sel.columns
    )

    assert not X_train_sel.isna().any().any()
    assert not X_val_sel.isna().any().any()

    return (
        features_finales,
        X_train_sel,
        X_val_sel,
        modelo_reducido
    )


def transformar_test_final(
    test,
    frio_diario,
    features_filtradas,
    umbrales_extremos,
    medianas_train,
    columnas_indicadores,
    medianas_historicas,
    X_train_preparado,
):
    """Transforma las covariables de Test DESPUÉS de seleccionar el modelo.

    Nunca accede a target_manana ni aprende percentiles, medianas, banderas
    o nuevas columnas de Test. Los lags observan exclusivamente días previos.
    """
    # Mismas columnas y orden definidos en Train.
    X_test = test[features_filtradas].copy()
    X_test_limpio = X_test.mask(
        X_test.gt(umbrales_extremos, axis="columns")
    )

    X_test_final = X_test_limpio.fillna(medianas_train).copy()
    indicadores = X_test_limpio[columnas_indicadores].isna().astype(int)
    indicadores.columns = [f"{c} | faltante" for c in columnas_indicadores]
    X_test_final = pd.concat([X_test_final, indicadores], axis=1)

    X_test_cal = agregar_calendario(X_test_final, test["fecha"])

    # Se utiliza la misma definición de lags y medias pasadas del entrenamiento.
    serie_frio = (
        frio_diario.set_index("fecha")["frio_kw"].sort_index().asfreq("D")
    )
    historicas = pd.DataFrame(index=serie_frio.index)
    historicas["frio_lag_1d"] = serie_frio.shift(1)
    historicas["frio_lag_7d"] = serie_frio.shift(7)
    historicas["frio_media_3d"] = (
        serie_frio.shift(1).rolling(3, min_periods=3).mean()
    )
    historicas["frio_media_7d"] = (
        serie_frio.shift(1).rolling(7, min_periods=7).mean()
    )
    historicas = historicas.reset_index()
    columnas_historicas = [
        "frio_lag_1d", "frio_lag_7d", "frio_media_3d", "frio_media_7d"
    ]
    valores = test[["fecha"]].merge(
        historicas, on="fecha", how="left", validate="one_to_one"
    )
    X_test_hist = X_test_cal.copy()
    for columna in columnas_historicas:
        X_test_hist[columna] = valores[columna].to_numpy()

    X_test_preparado = X_test_hist.copy()
    for columna in columnas_historicas:
        X_test_preparado[f"{columna} | faltante"] = (
            X_test_hist[columna].isna().astype(int)
        )
        X_test_preparado[columna] = X_test_hist[columna].fillna(
            medianas_historicas[columna]
        )

    assert X_test_limpio.shape == (159, 87)
    assert X_test_preparado.shape == (159, 144)
    assert X_test_preparado.columns.equals(X_train_preparado.columns)
    assert not X_test_preparado.isna().any().any()

    print("\nTEST: TRANSFORMACIÓN POSTERIOR A LA SELECCIÓN")
    print("Matriz inicial:", X_test_limpio.shape)
    print("Matriz preparada:", X_test_preparado.shape)
    print("No se consultaron etiquetas ni se recalcularon parámetros con Test.")
    return X_test_limpio, X_test_preparado


def transformar_derivadas_test(
    X_test_ampliado,
    nuevas_features,
    medianas_derivadas,
    umbrales_derivadas,
    indicadores_derivadas
):
    """
    Aplica a Test las reglas de limpieza e imputación
    aprendidas exclusivamente con Train.

    No ajusta ningún parámetro nuevo.
    """

    resultado = X_test_ampliado.copy()

    columnas_eficiencia = [
        c for c in nuevas_features
        if c.startswith("Eficiencia | ")
    ]

    columnas_ratios_interacciones = [
        c for c in nuevas_features
        if c.startswith(("Ratio | ", "Interaccion | "))
    ]

    # 1. Reemplazar infinitos
    resultado[nuevas_features] = (
        resultado[nuevas_features]
        .replace([np.inf, -np.inf], np.nan)
    )

    # 2. Aplicar las mismas reglas de validez
    for columna in columnas_eficiencia:
        resultado[columna] = resultado[columna].mask(
            resultado[columna] < -100
        )

    for columna in columnas_ratios_interacciones:
        resultado[columna] = resultado[columna].mask(
            resultado[columna] < 0
        )

    # 3. Aplicar umbrales aprendidos con Train
    mascara_extremos = resultado[nuevas_features].gt(
        umbrales_derivadas,
        axis="columns"
    )

    resultado[nuevas_features] = (
        resultado[nuevas_features]
        .mask(mascara_extremos)
    )

    # 4. Crear los mismos indicadores definidos en Train
    for columna in indicadores_derivadas:
        resultado[f"{columna} | faltante"] = (
            resultado[columna].isna().astype(int)
        )

    # 5. Imputar con medianas de Train
    resultado[nuevas_features] = (
        resultado[nuevas_features]
        .fillna(medianas_derivadas)
    )

    return resultado


def exportar_dataset_final(
    train,
    val,
    test,
    X_train_sel,
    X_val_sel,
    X_test_sel,
    features_finales
):
    """
    Exporta el dataset procesado con:
    - 22 características seleccionadas.
    - Objetivo del día siguiente.
    - Fechas y procedencia.
    - Identificación de partición.

    No aplica escalado.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    columnas_metadata = [
        "fecha",
        "fecha_siguiente",
        "archivo_origen",
        "target_manana"
    ]

    # 1. Unir metadatos y características
    # sin depender de los índices originales
    def preparar_particion(datos, X, nombre):

        assert len(datos) == len(X)
        assert X.columns.tolist() == features_finales

        metadata = (
            datos[columnas_metadata]
            .reset_index(drop=True)
            .copy()
        )

        caracteristicas = (
            X.reset_index(drop=True)
            .copy()
        )

        resultado = pd.concat(
            [metadata, caracteristicas],
            axis=1
        )

        resultado.insert(
            0,
            "particion",
            nombre
        )

        return resultado

    dataset_train = preparar_particion(
        train, X_train_sel, "train"
    )

    dataset_val = preparar_particion(
        val, X_val_sel, "validation"
    )

    dataset_test = preparar_particion(
        test, X_test_sel, "test"
    )

    # 2. Unificar en orden cronológico
    dataset_final = pd.concat(
        [dataset_train, dataset_val, dataset_test],
        ignore_index=True
    )

    # 3. Ordenar y comprobar fechas
    dataset_final = (
        dataset_final
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    assert (
        dataset_final["fecha_siguiente"]
        == dataset_final["fecha"] + pd.Timedelta(days=1)
    ).all()

    # 4. Comprobar integridad
    assert dataset_final.shape == (1052, 27)
    assert dataset_final["fecha"].is_unique
    assert dataset_final.columns.is_unique

    assert (
        dataset_final[features_finales]
        .notna().all().all()
    )

    assert np.isfinite(
        dataset_final[features_finales].to_numpy(
            dtype=float
        )
    ).all()

    assert (
        dataset_final["target_manana"]
        .notna().all()
    )

    # 5. Guardar el archivo sin escalado
    ruta_salida = (
        PROCESSED_DIR / "dataset_final.csv"
    )

    dataset_final.to_csv(
        ruta_salida,
        index=False,
        encoding="utf-8",
        date_format="%Y-%m-%d",
        lineterminator="\n"
    )

    # 6. Mostrar solamente controles estructurales
    # No evaluar predicciones ni errores de Test
    print("\nEXPORTACIÓN DEL DATASET FINAL")

    print("Ruta:", ruta_salida.relative_to(RAIZ))
    print("Dimensiones:", dataset_final.shape)

    print("\nObservaciones por partición:")
    print(
        dataset_final["particion"]
        .value_counts()
        .reindex(["train", "validation", "test"])
        .to_string()
    )

    print(
        "\nCaracterísticas seleccionadas:",
        len(features_finales)
    )

    print(
        "Valores faltantes:",
        int(dataset_final.isna().sum().sum())
    )

    print(
        "Fechas duplicadas:",
        int(dataset_final["fecha"].duplicated().sum())
    )

    print("\nDataset exportado sin escalado.")

    return dataset_final, ruta_salida


def generar_data_lineage(
    dataset_final,
    features_finales,
    train,
    val,
    test,
    columnas_descartadas,
    medianas_train,
    umbrales_extremos,
    medianas_historicas,
    medianas_derivadas,
    umbrales_derivadas,
    indicadores_derivadas,
    mejor_lasso
):
    """
    Registra fuentes, transformaciones, parámetros,
    particiones y características del dataset final.
    """

    fecha_ejecucion = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds")

    # Convertir parámetros de pandas a JSON
    def convertir_serie(serie):
        return {
            str(columna): float(valor)
            for columna, valor in serie.dropna().items()
        }

    # Registrar cada transformación y su justificación
    transformaciones = [
        {
            "funcion": "limpiar_temporal",
            "decision": (
                "Excluir 615 registros sin referencia temporal "
                "y corregir 587 desfases exactos de un segundo."
            )
        },
        {
            "funcion": "seleccionar_jornadas_temporales",
            "decision": (
                "Excluir 16 fechas ambiguas con jornadas apiladas. "
                "Exigir 25 horarios y cierre a las 23:59."
            )
        },
        {
            "funcion": "controlar_calidad_frio",
            "decision": (
                "Excluir jornadas con reinicios incorrectos, "
                "descensos, nulos, negativos o valores extremos."
            )
        },
        {
            "funcion": "consolidar_cierres",
            "decision": (
                "Resolver solapamientos priorizando la fuente "
                "más reciente cuando coincide el cierre. "
                "Excluir el conflicto del 2022-02-02."
            )
        },
        {
            "funcion": "construir_objetivo",
            "decision": (
                "Definir target_manana solo para fechas "
                "calendario consecutivas D y D+1."
            )
        },
        {
            "funcion": "integrar_electricidad / integrar_produccion / "
                       "integrar_mycom / integrar_agua / "
                       "integrar_area_consolidada / integrar_kpi",
            "decision": (
                "Integrar las áreas por fecha y archivo de origen, "
                "utilizando cierres de jornadas completas."
            )
        },
        {
            "funcion": "dividir_temporal",
            "decision": (
                "Separación cronológica 70/15/15, "
                "descartando dos observaciones de frontera."
            )
        },
        {
            "funcion": "filtrar_variables_faltantes / "
                       "tratar_valores_extremos / imputar_faltantes",
            "decision": (
                "Filtrar columnas con más del 50 % de faltantes, "
                "aplicar umbrales de 10 veces P99 e imputar "
                "con medianas calculadas solo en Train."
            )
        },
        {
            "funcion": "agregar_calendario / agregar_historicas / "
                       "imputar_historicas",
            "decision": (
                "Agregar estacionalidad, lags y medias móviles "
                "sin utilizar consumos posteriores al día D."
            )
        },
        {
            "funcion": "crear_variables_eficiencia / "
                       "calcular_ratios_industriales / "
                       "calcular_interacciones_industriales",
            "decision": (
                "Construir características derivadas; ratios e "
                "interacciones usan valores base depurados."
            )
        },
        {
            "funcion": "preparar_caracteristicas_derivadas",
            "decision": (
                "Tratar anomalías e imputar derivadas usando "
                "únicamente parámetros aprendidos en Train."
            )
        },
        {
            "funcion": "seleccionar_variables_rf / "
                       "seleccionar_variables_lasso / "
                       "verificar_seleccion_final",
            "decision": (
                "Comparar Random Forest y LASSO; elegir "
                "22 características utilizando Train y Validation."
            )
        },
        {
            "funcion": "exportar_dataset_final",
            "decision": (
                "Exportar el dataset procesado sin escalado, "
                "con objetivo, metadatos y partición."
            )
        }
    ]

    # Añadir script y fecha a cada transformación
    for paso in transformaciones:
        paso["script"] = "src/preprocessing_pipeline.py"
        paso["fecha_ejecucion_utc"] = fecha_ejecucion

    # Registrar las fuentes y su contenido
    fuentes = {
        f"data/{archivo.name}": calcular_sha256(archivo)
        for archivo in ARCHIVOS
    }

    # Registrar límites de cada partición
    particiones = {}

    for nombre, datos in [
        ("train", train),
        ("validation", val),
        ("test", test)
    ]:
        particiones[nombre] = {
            "filas": len(datos),
            "fecha_inicio": str(datos["fecha"].min().date()),
            "fecha_fin": str(datos["fecha"].max().date())
        }

    linaje = {
        "fecha_generacion_utc": fecha_ejecucion,
        "script": "src/preprocessing_pipeline.py",
        "fuentes_sha256": fuentes,

        "dataset_final": {
            "archivo": "data/processed/dataset_final.csv",
            "sha256": calcular_sha256(
                PROCESSED_DIR / "dataset_final.csv"
            ),
            "filas": len(dataset_final),
            "columnas": len(dataset_final.columns),
            "escalado": False,
            "objetivo": "target_manana",
            "horizonte": "D+1",
            "caracteristicas": features_finales
        },

        "particiones": particiones,

        "parametros_train": {
            "umbral_faltantes": 0.50,
            "columnas_descartadas": (
                columnas_descartadas.index.tolist()
            ),
            "factor_extremos": 10,
            "medianas_industriales": convertir_serie(
                medianas_train
            ),
            "umbrales_industriales": convertir_serie(
                umbrales_extremos
            ),
            "medianas_historicas": convertir_serie(
                medianas_historicas
            ),
            "medianas_derivadas": convertir_serie(
                medianas_derivadas
            ),
            "umbrales_derivadas": convertir_serie(
                umbrales_derivadas
            ),
            "indicadores_derivadas": indicadores_derivadas
        },

        "seleccion": {
            "metodos": ["Random Forest", "LASSO"],
            "metodo_final": "LASSO",
            "alpha": float(
                mejor_lasso.named_steps["lasso"].alpha
            ),
            "numero_caracteristicas": len(features_finales),
            "criterio": "RMSE en Validation"
        },

        "transformaciones": transformaciones
    }

    # Guardar JSON
    ruta_linaje = (
        PROCESSED_DIR / "data_lineage.json"
    )

    with ruta_linaje.open(
        "w", encoding="utf-8"
    ) as archivo:
        json.dump(
            linaje,
            archivo,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

    print("\nDATA LINEAGE")
    print("Archivo:", ruta_linaje.relative_to(RAIZ))
    print("Transformaciones:", len(transformaciones))
    print("Fuentes registradas:", len(fuentes))
    print("Características:", len(features_finales))
    print("Parámetros de Train: guardados")
    print("SHA-256 del CSV: registrado")

    assert ruta_linaje.is_file()

    return ruta_linaje



def registrar_checksum_dataset(ruta_dataset):
    """
    Registra el SHA-256 del dataset procesado.

    Conserva los checksums existentes y verifica
    su coincidencia con data_lineage.json.
    """

    ruta_checksums = DATA_DIR / "checksums.json"
    ruta_lineage = PROCESSED_DIR / "data_lineage.json"

    # 1. Leer el manifiesto existente
    with ruta_checksums.open(encoding="utf-8") as f:
        checksums = json.load(f)

    if not isinstance(checksums, dict):
        raise ValueError(
            "El manifiesto de checksums debe ser un diccionario."
        )

    entradas_anteriores = checksums.copy()

    # 2. Calcular el SHA-256 del dataset final
    clave = ruta_dataset.relative_to(RAIZ).as_posix()

    hash_dataset = calcular_sha256(ruta_dataset)

    # 3. Verificar coincidencia con el linaje
    with ruta_lineage.open(encoding="utf-8") as f:
        lineage = json.load(f)

    hash_lineage = lineage["dataset_final"]["sha256"]

    if hash_dataset != hash_lineage:
        raise ValueError(
            "El SHA-256 no coincide con data_lineage.json."
        )

    # 4. Agregar o actualizar únicamente esta entrada
    checksums[clave] = hash_dataset

    # 5. Guardar el manifiesto
    with ruta_checksums.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            checksums,
            f,
            indent=2,
            ensure_ascii=False
        )
        f.write("\n")

    # 6. Volver a leer para verificar
    with ruta_checksums.open(encoding="utf-8") as f:
        verificacion = json.load(f)

    assert verificacion[clave] == hash_dataset

    # Verificar que ninguna otra entrada cambió
    for nombre, valor in entradas_anteriores.items():
        if nombre != clave:
            assert verificacion[nombre] == valor

    print("\nREGISTRO DE CHECKSUMS")
    print("Archivo:", ruta_checksums.relative_to(RAIZ))
    print("Dataset registrado:", clave)
    print("SHA-256:", hash_dataset)

    print(
        "Entradas totales del manifiesto:",
        len(verificacion)
    )

    print("Coincidencia con data_lineage.json: OK")
    print("Entradas anteriores conservadas: OK")

    return hash_dataset






def main():
    verificar_fuentes()

    ee_raw = cargar_electricidad()

    print("\nCARGA ORIGINAL")
    print("Registros originales:", len(ee_raw))
    print("Fechas inválidas:", ee_raw["fecha"].isna().sum())
    print("Horas inválidas:", ee_raw["hora"].isna().sum())

    ee_trabajo = limpiar_temporal(ee_raw)

    assert len(ee_trabajo) == 42900

    ee_temporal = seleccionar_jornadas_temporales(
        ee_trabajo
    )

    assert len(ee_temporal) == 40650

    # Nuevo paso: controlar calidad del totalizador
    cierres_aprobados, calidad_jornadas = (
        controlar_calidad_frio(ee_temporal)
    )

    print("\nControl de calidad completado.")
     # Resolver solapamientos entre los tres Excel
    frio_diario, conflictos = consolidar_cierres(
        cierres_aprobados
    )

    print("\nConsolidación de refrigeración completada.")

    # Construir objetivo para el día siguiente
    dataset_base = construir_objetivo(frio_diario)

    print("\nConstrucción del objetivo completada.")


    # Integrar variables eléctricas del día D
    dataset_integrado = integrar_electricidad(
        dataset_base,
        ee_temporal
    )

    print("\nIntegración eléctrica completada.")

    # Integrar la actividad productiva del día D
    dataset_integrado = integrar_produccion(
        dataset_integrado,
        dataset_base
    )

    print("\nIntegración productiva completada.")

     # Integrar los siete compresores Mycom
    dataset_integrado = integrar_mycom(
        dataset_integrado,
        dataset_base
    )

    print("\nIntegración de Mycom completada.")

    # Integrar consumos de agua del día D
    dataset_integrado = integrar_agua(
        dataset_integrado,
        dataset_base
    )

    print("\nIntegración de agua completada.")

    # Integrar Gas y Vapor
    dataset_integrado = integrar_area_consolidada(
        dataset_integrado,
        dataset_base,
        hoja="Consolidado GasVapor",
        prefijo="GasVapor"
    )

    # Verificar contra los resultados del notebook
    assert len([
        c for c in dataset_integrado.columns
        if c.startswith("GasVapor | ")
    ]) == 19

    assert dataset_integrado.shape == (1126, 80)

    print("\nIntegración de GasVapor completada.")


    # Integrar consumos de aire comprimido
    dataset_integrado = integrar_area_consolidada(
        dataset_integrado,
        dataset_base,
        hoja="Consolidado Aire",
        prefijo="Aire"
    )

    # Verificar las variables del área
    columnas_aire = [
        c for c in dataset_integrado.columns
        if c.startswith("Aire | ")
    ]

    print("\nVERIFICACIÓN FINAL DE INTEGRACIÓN")
    print("Variables de Aire:", len(columnas_aire))
    print("Dimensiones:", dataset_integrado.shape)

    # Validar contra el notebook
    assert len(columnas_aire) == 12
    assert dataset_integrado.shape == (1126, 92)
    assert dataset_integrado["fecha"].is_unique

    assert (
        dataset_integrado["target_manana"].notna().sum()
        == 1054
    )

    print("\nIntegración industrial completada.")


    # División cronológica sin solapamiento temporal
    train, val, test, features = dividir_temporal(
        dataset_integrado
    )

    print("\nDivisión temporal completada.")

    # Preparación exclusiva de Train y Validation.
    # Test se reserva hasta haber concluido la selección de características.
    (
        X_train,
        X_val,
        features_filtradas,
        columnas_descartadas
    ) = filtrar_variables_faltantes(train, val, features)
    print("\nFiltrado de características completado.")

    (
        X_train_limpio,
        X_val_limpio,
        umbrales_extremos,
        resumen_extremos
    ) = tratar_valores_extremos(X_train, X_val)
    print("\nTratamiento de extremos completado.")

    (
        X_train_final,
        X_val_final,
        medianas_train,
        columnas_indicadores
    ) = imputar_faltantes(X_train_limpio, X_val_limpio)
    print("\nImputación completada.")

    X_train_cal = agregar_calendario(X_train_final, train["fecha"])
    X_val_cal = agregar_calendario(X_val_final, val["fecha"])

    print("\nINGENIERÍA DE CALENDARIO")
    print("Variables añadidas: 5")
    print("Train:", X_train_cal.shape)
    print("Validation:", X_val_cal.shape)
    assert X_train_cal.shape == (736, 136)
    assert X_val_cal.shape == (157, 136)
    assert X_train_cal.columns.equals(X_val_cal.columns)
    assert not X_train_cal.isna().any().any()
    assert not X_val_cal.isna().any().any()
    print("\nCaracterísticas de calendario incorporadas.")

    X_train_hist, X_val_hist = agregar_historicas(
        X_train_cal, X_val_cal, train, val, frio_diario
    )
    print("\nVariables históricas incorporadas.")

    (
        X_train_preparado,
        X_val_preparado,
        medianas_historicas
    ) = imputar_historicas(X_train_hist, X_val_hist)
    print("\nIngeniería histórica completada.")


    # Integrar indicadores KPI y metas energéticas
    dataset_kpi = integrar_kpi(
        dataset_integrado,
        dataset_base
    )

    print("\nIntegración de KPI completada.")

    # Calcular eficiencia respecto de metas KPI
    (
        dataset_eficiencia,
        columnas_eficiencia
    ) = crear_variables_eficiencia(
        dataset_kpi
    )

    print("\nVariables de eficiencia incorporadas.")


    # Crear ratios usando variables base depuradas,
    # antes de su imputación
    ratios_train = calcular_ratios_industriales(
        X_train_limpio
    )

    ratios_val = calcular_ratios_industriales(
        X_val_limpio
    )

    print("\nRATIOS INDUSTRIALES")
    print("Ratios añadidos:", ratios_train.shape[1])

    print("\nDIMENSIONES")
    print("Train:", ratios_train.shape)
    print("Validation:", ratios_val.shape)

    print("\nFALTANTES POR RATIO")

    resumen_ratios = pd.DataFrame({
        "Train": ratios_train.isna().sum(),
        "Validation": ratios_val.isna().sum()
    })

    print(resumen_ratios.to_string())

    # Verificar alineación con las particiones
    assert ratios_train.shape == (736, 4)
    assert ratios_val.shape == (157, 4)

    assert ratios_train.index.equals(
        X_train_preparado.index
    )

    assert ratios_val.index.equals(
        X_val_preparado.index
    )

    assert ratios_train.columns.equals(
        ratios_val.columns
    )

    print("\nRatios industriales calculados.")


    # Crear interacciones con variables depuradas
    interacciones_train = calcular_interacciones_industriales(
        X_train_limpio
    )

    interacciones_val = calcular_interacciones_industriales(
        X_val_limpio
    )

    print("\nINTERACCIONES INDUSTRIALES")
    print("Interacciones creadas:", interacciones_train.shape[1])

    print("\nDIMENSIONES")
    print("Train:", interacciones_train.shape)
    print("Validation:", interacciones_val.shape)

    print("\nFALTANTES POR INTERACCIÓN")

    resumen_interacciones = pd.DataFrame({
        "Train": interacciones_train.isna().sum(),
        "Validation": interacciones_val.isna().sum()
    })

    print(resumen_interacciones.to_string())

    # Validaciones
    assert interacciones_train.shape == (736, 3)
    assert interacciones_val.shape == (157, 3)

    assert interacciones_train.index.equals(
        X_train_preparado.index
    )

    assert interacciones_val.index.equals(
        X_val_preparado.index
    )

    assert interacciones_train.columns.equals(
        interacciones_val.columns
    )

    print("\nInteracciones industriales calculadas.")


    # Unir las características derivadas con Train
    X_train_ampliado, nuevas_features = (
        unir_caracteristicas_derivadas(
            X_train_preparado,
            train,
            dataset_eficiencia,
            columnas_eficiencia,
            ratios_train,
            interacciones_train
        )
    )

    # Aplicar la misma estructura a Validation
    X_val_ampliado, _ = (
        unir_caracteristicas_derivadas(
            X_val_preparado,
            val,
            dataset_eficiencia,
            columnas_eficiencia,
            ratios_val,
            interacciones_val
        )
    )

    print("\nUNIÓN DE CARACTERÍSTICAS DERIVADAS")

    print("Nuevas características:", len(nuevas_features))
    print("Train:", X_train_ampliado.shape)
    print("Validation:", X_val_ampliado.shape)

    print("\nFALTANTES DE LAS NUEVAS VARIABLES")

    resumen = pd.DataFrame({
        "Train": X_train_ampliado[
            nuevas_features
        ].isna().sum(),
        "Validation": X_val_ampliado[
            nuevas_features
        ].isna().sum()
    })

    print(resumen.to_string())

    assert X_train_ampliado.columns.equals(
        X_val_ampliado.columns
    )

    print("\nUnión de características completada.")


    # Limpiar e imputar las características derivadas
    (
        X_train_completo,
        X_val_completo,
        medianas_derivadas,
        umbrales_derivadas,
        indicadores_derivadas
    ) = preparar_caracteristicas_derivadas(
        X_train_ampliado,
        X_val_ampliado,
        nuevas_features
    )

    print("\nPreparación de características completada.")


    # Selección de variables con Random Forest
    # El objetivo corresponde al consumo del día D+1
    y_train = train["target_manana"].to_numpy()

    (
        features_rf,
        importancias_rf,
        selector_rf
    ) = seleccionar_variables_rf(
        X_train_completo,
        y_train
    )

    print("\nSelección Random Forest completada.")


    # Objetivo de validación
    y_val = val["target_manana"].to_numpy()

    # Segundo método: LASSO
    (
        features_lasso,
        coeficientes_lasso,
        mejor_lasso,
        tabla_lasso
    ) = seleccionar_variables_lasso(
        X_train_completo,
        y_train,
        X_val_completo,
        y_val
    )

    # Comparar subconjuntos y modelos de referencia
    (
        tabla_comparacion,
        coincidentes,
        union
    ) = comparar_selecciones(
        X_train_completo,
        y_train,
        X_val_completo,
        y_val,
        features_rf,
        features_lasso,
        mejor_lasso
    )

    print("\nSelección y comparación completadas.")


    # Verificar que las 22 variables LASSO
    # sean suficientes para el modelo reducido
    (
        features_finales,
        X_train_sel,
        X_val_sel,
        modelo_lasso_reducido
    ) = verificar_seleccion_final(
        X_train_completo,
        y_train,
        X_val_completo,
        y_val,
        features_lasso,
        mejor_lasso
    )

    print("\nVerificación final de características completada.")


    # ==========================================
    # PREPARAR TEST SIN UTILIZAR SU OBJETIVO
    # ==========================================

    # Solo ahora, tras fijar alpha y features_finales,
    # se transforman las covariables de Test (sin consultar su objetivo).
    X_test_limpio, X_test_preparado = transformar_test_final(
        test,
        frio_diario,
        features_filtradas,
        umbrales_extremos,
        medianas_train,
        columnas_indicadores,
        medianas_historicas,
        X_train_preparado
    )

    # 1. Ratios e interacciones con datos depurados
    ratios_test = calcular_ratios_industriales(
        X_test_limpio
    )

    interacciones_test = calcular_interacciones_industriales(
        X_test_limpio
    )

    # 2. Incorporar las 11 características derivadas
    X_test_ampliado, features_derivadas_test = (
        unir_caracteristicas_derivadas(
            X_test_preparado,
            test,
            dataset_eficiencia,
            columnas_eficiencia,
            ratios_test,
            interacciones_test
        )
    )

    assert features_derivadas_test == nuevas_features

    # 3. Aplicar reglas aprendidas con Train
    X_test_completo = transformar_derivadas_test(
        X_test_ampliado,
        nuevas_features,
        medianas_derivadas,
        umbrales_derivadas,
        indicadores_derivadas
    )

    # 4. Conservar únicamente las 22 variables
    X_test_sel = X_test_completo[
        features_finales
    ].copy()

    # 5. Comprobar la estructura
    print("\nPREPARACIÓN DE TEST")

    print("Características iniciales:", X_test_preparado.shape[1])
    print("Después de derivadas:", X_test_completo.shape)
    print("Después de selección:", X_test_sel.shape)

    print(
        "Faltantes restantes:",
        int(X_test_sel.isna().sum().sum())
    )

    assert X_test_ampliado.shape == (159, 155)
    assert X_test_completo.shape == (159, 163)
    assert X_test_sel.shape == (159, 22)

    assert X_test_completo.columns.equals(
        X_train_completo.columns
    )

    assert X_test_sel.columns.equals(
        X_train_sel.columns
    )

    assert not X_test_sel.isna().any().any()

    print("\nTest preparado sin evaluar predicciones.")


    # Exportar las tres particiones con las
    # 22 características seleccionadas
    dataset_final, ruta_dataset_final = (
        exportar_dataset_final(
            train,
            val,
            test,
            X_train_sel,
            X_val_sel,
            X_test_sel,
            features_finales
        )
    )

    print("\nExportación completada.")


    # Registrar el linaje del dataset procesado
    ruta_linaje = generar_data_lineage(
        dataset_final,
        features_finales,
        train,
        val,
        test,
        columnas_descartadas,
        medianas_train,
        umbrales_extremos,
        medianas_historicas,
        medianas_derivadas,
        umbrales_derivadas,
        indicadores_derivadas,
        mejor_lasso
    )

    print("\nLinaje de datos generado correctamente.")


    # Registrar checksum del dataset final
    checksum_final = registrar_checksum_dataset(
        ruta_dataset_final
    )

    print("\nVersionado de datos completado.")










    













if __name__ == "__main__":
    main()
