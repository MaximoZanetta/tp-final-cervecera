# Predicción de Consumo Energético Industrial

**Trabajo Práctico Final 2026 — Laboratorio de Datos II**

## Objetivo

Desarrollar un modelo de aprendizaje automático capaz de predecir el consumo energético diario del sistema de refrigeración de una planta cervecera para el día siguiente (D+1).

Para ello se utilizan registros históricos de totalizadores industriales correspondientes al período 2020–2023, incluyendo información de electricidad, producción, refrigeración, agua, aire comprimido, gas, vapor e indicadores de eficiencia energética.

## Tecnologías

- Python 3.12
- Pandas y NumPy
- Scikit-learn
- Matplotlib y Seaborn
- Jupyter Notebook
- Git y GitHub

Se contempla utilizar XGBoost y LightGBM en las etapas posteriores de modelado.

## Instalación

Crear y activar el entorno de trabajo:

```bash
conda create -n cervecera_env python=3.12
conda activate cervecera_env
python -m pip install uv
uv pip install -r requirements.txt
```

## Datos

Los archivos Excel originales no se incluyen en el repositorio y deben colocarse dentro de la carpeta `data/`, respetando exactamente estos nombres:

- `Totalizadores Planta 2020_2022.xlsx`
- `Totalizadores Planta - 2021_2023.xlsx`
- `Totalizadores Planta 2022_2023.xlsx`

El archivo `data/checksums.json` contiene los hashes SHA-256 utilizados para comprobar la integridad de las fuentes y del dataset procesado.

Para reproducir el preprocesamiento desde los datos originales, es necesario contar con los tres Excel.

## Estructura del proyecto

```text
tp-final/
├── data/
│   ├── processed/
│   │   ├── dataset_final.csv
│   │   └── data_lineage.json
│   └── checksums.json
├── notebooks/
│   ├── eda.ipynb
│   └── preprocesamiento.ipynb
├── src/
│   └── preprocessing_pipeline.py
├── requirements.txt
├── .gitignore
└── README.md
```

Los nombres de los notebooks de otras fases pueden variar.

## Fase 1: Análisis Exploratorio de Datos

Se analizaron los registros históricos de consumo energético industrial para identificar patrones, distribuciones, valores faltantes, registros anómalos e inconsistencias temporales.

Los resultados del EDA se utilizaron para justificar las decisiones de limpieza y construcción de características de la Fase 2.

## Fase 2: Preprocesamiento y Selección de Variables

El procesamiento incluye:

1. **Limpieza:** control de jornadas incompletas, reinicios incorrectos de totalizadores, descensos anómalos y valores extremos.
2. **Integración:** consolidación de los registros diarios provenientes de las distintas áreas industriales.
3. **Tratamiento de faltantes:** eliminación de variables con exceso de valores ausentes, imputación mediante medianas de entrenamiento e incorporación de indicadores de ausencia.
4. **Ingeniería de características:** variables de calendario, lags, promedios móviles, ratios entre consumos, indicadores de eficiencia respecto de metas e interacciones.
5. **Selección de variables:** comparación entre importancia de características con Random Forest y regularización LASSO.
6. **Preparación final:** generación del dataset procesado sin aplicar escalado previo.

Los parámetros de limpieza e imputación se calculan usando Train y se reutilizan en las demás particiones.

### División temporal

Los datos se dividen cronológicamente, sin mezclarlos aleatoriamente:

| Conjunto | Observaciones | Proporción aproximada |
|---|---:|---:|
| Train | 736 | 70 % |
| Validation | 157 | 15 % |
| Test | 159 | 15 % |
| **Total** | **1.052** | **100 %** |

Se eliminaron dos observaciones en las fronteras temporales para evitar solapamientos entre fechas objetivo y fechas de las particiones siguientes.

### Selección de variables

Se evaluaron 163 características mediante Random Forest y LASSO. Se seleccionó un conjunto final de **22 variables** utilizando el rendimiento en Validation.

El modelo LASSO reducido, con `alpha=300`, obtuvo:

- RMSE de Validation: **2.961,56**
- MAE de Validation: **2.365,00**

Estos valores corresponden al proceso de selección y no representan una evaluación final independiente.

El conjunto de Test se reserva para la evaluación final del modelo.

### Escalado

El archivo procesado se guarda sin normalizar ni estandarizar.

Cuando el modelo requiere escalado, se utiliza `StandardScaler` dentro de un `Pipeline` de scikit-learn, ajustado exclusivamente con los datos de entrenamiento.

## Ejecución del preprocesamiento

Desde la raíz del proyecto, con el entorno activado:

```bash
python src/preprocessing_pipeline.py
```

El script verifica las fuentes originales, ejecuta las transformaciones, prepara las particiones temporales y genera los entregables.

También se puede consultar y ejecutar el procedimiento documentado en:

`notebooks/preprocesamiento.ipynb`

Para ejecutarlo en VS Code, seleccionar el kernel `cervecera_env` y utilizar **Restart Kernel and Run All**, después de generar los datos mediante el script.

## Archivos generados

**`data/processed/dataset_final.csv`**

Dataset procesado sin escalar, con 1.052 observaciones y 27 columnas: 22 características, el objetivo y cuatro columnas adicionales de identificación y metadatos.

**`data/processed/data_lineage.json`**

Registro de las fuentes originales, transformaciones realizadas, fechas, parámetros de entrenamiento, particiones y características seleccionadas.

**`data/checksums.json`**

Registro de hashes SHA-256 para controlar la integridad de los archivos.

## Reproducibilidad

El procesamiento utiliza particiones cronológicas, reglas de transformación definidas a partir de Train y semillas fijas para los métodos de selección que involucran aleatoriedad.

El dataset generado puede verificarse mediante su checksum SHA-256, registrado tanto en `data/checksums.json` como en `data/processed/data_lineage.json`.

## Próximas etapas

Las siguientes fases abordarán el entrenamiento y comparación de modelos predictivos, el ajuste de hiperparámetros y la evaluación final sobre el conjunto de Test.
