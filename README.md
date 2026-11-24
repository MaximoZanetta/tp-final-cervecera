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

Para el entrenamiento y la optimización de modelos también se utilizan XGBoost, LightGBM y Optuna. Los modelos entrenados se serializan mediante Joblib y se documentan en un registro local de modelos.

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

## Fase 3: Entrenamiento y Versionado de Modelos

Se entrenaron y compararon diferentes modelos de regresión para predecir el consumo de refrigeración del día siguiente, utilizando las 22 características seleccionadas durante el preprocesamiento.

### Modelos evaluados

Se implementaron dos modelos de referencia obligatorios:

- **Persistencia:** predice para mañana el mismo consumo observado hoy.
- **Media móvil de 7 días:** utiliza el promedio del consumo de los siete días calendario hasta el día actual. Cuando hay fechas faltantes, considera los días disponibles.

Posteriormente se entrenaron cinco alternativas:

- Ridge Regression.
- LASSO Regression.
- Random Forest Regressor.
- XGBoost.
- LightGBM.

Para Random Forest, XGBoost y LightGBM se utilizó Optuna con `TPESampler` y semilla 42 para optimizar los hiperparámetros.

La selección del modelo se realizó minimizando el MAE sobre Validation, manteniendo Test reservado para la evaluación final.

### Resultados en Validation

| Modelo | MAE | RMSE | R² |
|---|---:|---:|---:|
| LASSO | 2365.00 | 2961.56 | 0.447 |
| Media móvil de 7 días | 2492.70 | 3282.50 | 0.321 |
| Ridge | 2494.85 | 3087.42 | 0.399 |
| Random Forest | 2540.33 | 3110.43 | 0.390 |
| XGBoost | 2581.72 | 3144.04 | 0.377 |
| LightGBM | 2607.13 | 3231.81 | 0.342 |
| Persistencia | 2739.85 | 3419.28 | 0.263 |

El modelo seleccionado fue **LASSO con alpha = 300**, al obtener el menor MAE de validación.

La mejora respecto del mejor baseline fue aproximadamente del 5,1 %.

### Diagnóstico del modelo

Se realizó un análisis de residuos, distribución de errores, evolución temporal, comparación de predicciones frente a valores reales e interpretación de los coeficientes del modelo.

El análisis identificó una tendencia moderada a sobreestimar el consumo, especialmente en algunos días con consumos reales reducidos.

Los resultados y gráficos se encuentran en `results/` y se documentan en `notebooks/modelado.ipynb`.

### Registro y versionado

Una vez seleccionado el mejor modelo, LASSO se reentrenó con Train + Validation, utilizando 893 observaciones y 22 características.

Se generaron los siguientes archivos:

- `models/modelo_v1.0.0.pkl`: modelo serializado con StandardScaler y LASSO.
- `models/model_registry.json`: versión, hiperparámetros, métricas de validación, características, fecha y commit de entrenamiento.
- `results/experiment_logs.csv`: seguimiento de experimentos con hiperparámetros y métricas.

Las métricas de Validation registradas corresponden al modelo entrenado exclusivamente con Train, antes del reentrenamiento final.

### Ejecución

Para ejecutar los experimentos:

```bash
python src/train_model.py
```

Para generar el diagnóstico del modelo seleccionado:

```bash
python src/diagnostico_validacion.py
```

Para reentrenar y registrar el modelo ganador a partir del historial de experimentos existente:

```bash
python src/train_model.py --finalizar
```

El último comando requiere que el código de entrenamiento esté previamente versionado en Git.

El análisis también puede consultarse en `notebooks/modelado.ipynb`.

**El conjunto de Test todavía no fue utilizado para la evaluación del modelo.**

## Fase 4: Pipeline de predicción

Se implementó `src/predict.py`, que permite predecir el consumo eléctrico de refrigeración del día siguiente (D+1) a partir de un archivo Excel con mediciones industriales del día D.

### Ejecución

Instalar las dependencias:

```bash
uv pip install -r requirements.txt
```

Ejecutar el predictor desde la raíz del proyecto:

```bash
python src/predict.py "data/Totalizadores Planta - 2021_2023.xlsx"
```

El script carga automáticamente la versión del modelo indicada en `models/model_registry.json`, comprueba su integridad y reutiliza las transformaciones del preprocesamiento.

### Formato de entrada

El Excel debe contener las hojas necesarias para reconstruir las características del modelo:

- Consolidado EE
- Consolidado Produccion
- Consolidado Agua
- Consolidado GasVapor
- Consolidado Aire

Se permiten hojas y columnas adicionales, así como cambios en su orden. El script identifica las columnas por nombre y produce mensajes descriptivos cuando falta información requerida.

### Formato de salida

El resultado se guarda en:

`results/predicciones.csv`

Contiene dos columnas:

- `fecha`: fecha del día D.
- `prediccion_frio_kw`: consumo de refrigeración estimado para D+1.

Se genera una fila por cada jornada que supera los controles de calidad del preprocesamiento.

### Validaciones

Para verificar la estructura del Excel y la carga del modelo:

```bash
python src/predict.py "data/Totalizadores Planta - 2021_2023.xlsx" --validar
```

Para auditar los registros diarios:

```bash
python src/predict.py "data/Totalizadores Planta - 2021_2023.xlsx" --auditar
```

Para verificar las características generadas:

```bash
python src/predict.py "data/Totalizadores Planta - 2021_2023.xlsx" --caracteristicas
```

### Pruebas de robustez

Ejecutar:

```bash
python src/probar_robustez_predict.py
```

Se verificaron cinco casos:

1. Excel con hojas y columnas adicionales o reordenadas.
2. Falta de una columna obligatoria.
3. Falta de una hoja obligatoria.
4. Solicitud de una versión inexistente del modelo.
5. Generación de predicciones sobre datos sintéticos válidos.

Resultado: **5/5 pruebas correctas**.

### Resultados obtenidos

Con el Excel histórico `Totalizadores Planta - 2021_2023.xlsx`:

- Modelo: LASSO v1.0.0.
- Características utilizadas: 22.
- Días predichos: 568.
- Fechas duplicadas: 0.
- Predicciones no finitas: 0.

Estas pruebas verifican el funcionamiento del pipeline. No constituyen una evaluación del desempeño sobre el conjunto Test, que permanece reservado para la evaluación final
