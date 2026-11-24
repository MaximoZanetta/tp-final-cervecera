"""Evaluación final del modelo congelado sobre el Test interno de la Fase 2.

No entrena, no ajusta hiperparámetros y no modifica los artefactos oficiales.
Ejecutar desde la raíz: python src/evaluar_test.py
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from predict import load_model

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / 'data/processed/dataset_final.csv'
REGISTRO = RAIZ / 'models/model_registry.json'
CHECKSUMS = RAIZ / 'data/checksums.json'
VERSION = 'v1.0.0'


def main():
    # Verificar que el dataset sea el mismo con el que se entrenó el modelo.
    registro = json.loads(REGISTRO.read_text(encoding='utf-8'))['models'][VERSION]
    checksums = json.loads(CHECKSUMS.read_text(encoding='utf-8'))
    huella = hashlib.sha256(DATASET.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if huella != checksums['data/processed/dataset_final.csv'] or huella != registro['dataset_sha256_normalizado']:
        raise ValueError('El SHA-256 del dataset no coincide con los artefactos oficiales.')

    # Cargar el modelo ya entrenado y verificar su integridad desde el registry.
    modelo, features, version = load_model(VERSION)
    datos = pd.read_csv(DATASET, parse_dates=['fecha', 'fecha_siguiente'])
    esperado = ['train'] * 736 + ['validation'] * 157 + ['test'] * 159
    if datos['particion'].tolist() != esperado:
        raise ValueError('Las particiones no coinciden con la división temporal oficial.')
    if not datos['fecha'].is_monotonic_increasing or datos['fecha'].duplicated().any():
        raise ValueError('Las fechas no están ordenadas o contienen duplicados.')
    if not (datos['fecha_siguiente'] - datos['fecha']).eq(pd.Timedelta(days=1)).all():
        raise ValueError('El target no corresponde al día calendario siguiente.')
    test = datos.loc[datos['particion'].eq('test')].copy()
    X_test = test[features]
    y_test = test['target_manana'].to_numpy(dtype=float)

    # Misma definición que calcular_baselines() de train_model.py:
    # ventana calendario [D-6, D], incluye D, min_periods=1.
    # Usar el historial previo permite calcular las primeras ventanas de Test.
    serie = datos.set_index('fecha')['frio_kw'].sort_index().asfreq('D')
    media_7 = serie.rolling(window=7, min_periods=1).mean()
    predicciones = {
        'Persistencia': test['frio_kw'].to_numpy(dtype=float),
        'Media movil 7 dias': media_7.reindex(test['fecha']).to_numpy(),
    }
    if not np.isfinite(X_test.to_numpy(dtype=float)).all() or not np.isfinite(y_test).all():
        raise ValueError('Hay valores no finitos en el Test interno.')
    predicciones['LASSO'] = modelo.predict(X_test)

    filas = []
    for nombre, pred in predicciones.items():
        if len(pred) != len(test) or not np.isfinite(pred).all():
            raise ValueError(f'Predicciones inválidas de {nombre}.')
        filas.append({'modelo': nombre, 'particion': 'test', 'filas': len(test),
                      'MAE': mean_absolute_error(y_test, pred),
                      'RMSE': mean_squared_error(y_test, pred) ** 0.5,
                      'R2': r2_score(y_test, pred)})
    tabla = pd.DataFrame(filas)
    mae_baseline = tabla.loc[tabla['modelo'].ne('LASSO'), 'MAE'].min()
    mae_lasso = tabla.loc[tabla['modelo'].eq('LASSO'), 'MAE'].iloc[0]
    mejora = 100 * (mae_baseline - mae_lasso) / mae_baseline if mae_baseline else float('nan')
    detalle = test[['fecha', 'fecha_siguiente', 'target_manana']].copy()
    detalle['pred_persistencia'] = predicciones['Persistencia']
    detalle['pred_media_movil_7d'] = predicciones['Media movil 7 dias']
    detalle['pred_lasso'] = predicciones['LASSO']

    salida = RAIZ / 'results'
    salida.mkdir(exist_ok=True)
    tabla.to_csv(salida / 'metricas_test.csv', index=False)
    detalle.to_csv(salida / 'predicciones_test.csv', index=False)

    print('\nEVALUACION FINAL - TEST INTERNO')
    print('Integridad del dataset y del modelo: OK')
    print('Modelo:', version)
    print('Shape X_test:', X_test.shape)
    print('Periodo D:', test['fecha'].min().date(), 'a', test['fecha'].max().date())
    print(tabla.round(4).to_string(index=False))
    print(f'\nMejora frente al mejor baseline: {mejora:.2f} %')
    print('Mejora ambos baselines:', 'SI' if mae_lasso < mae_baseline else 'NO')
    print('MAE inferior a 3000 en Test interno:', 'SI' if mae_lasso < 3000 else 'NO')
    print('Resultados: results/metricas_test.csv y results/predicciones_test.csv')
    print('No se reentreno el modelo. El Test oculto externo sigue sin evaluar.')


if __name__ == '__main__':
    main()
