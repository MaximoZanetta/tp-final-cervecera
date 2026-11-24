"""Pruebas del contrato de entrada de src/predict.py (Fase 4).

Ejecutar desde la raíz del repositorio:
    python src/probar_robustez_predict.py

Crea archivos Excel sintéticos temporales, verifica hojas y columnas
extra/reordenadas, y los mensajes ante hojas/columnas ausentes.
También verifica la predicción de dos jornadas sintéticas con el
modelo registrado, sin consultar etiquetas reales ni Test.
"""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from openpyxl import Workbook

from predict import COLUMNAS_REQUERIDAS, load_model, predict_consumption, validar_entrada_excel


def crear_excel(ruta: Path, *, sin_hoja: str | None = None,
                sin_columna: tuple[str, str] | None = None) -> None:
    libro = Workbook()
    libro.remove(libro.active)
    for hoja, requeridas in COLUMNAS_REQUERIDAS.items():
        if hoja == sin_hoja:
            continue
        pagina = libro.create_sheet(hoja)
        # El orden de columnas difiere deliberadamente del Excel original.
        columnas = list(reversed(requeridas))
        if sin_columna and sin_columna[0] == hoja:
            columnas.remove(sin_columna[1])
        columnas.append("COLUMNA_EXTRA_NO_USADA")
        pagina.append(columnas)
        for d in (date(2024, 1, 8), date(2024, 1, 9)):
            horas = [f"{h:02d}:00:00" for h in range(24)] + ["23:59:00"]
            for indice, hora in enumerate(horas):
                fila = {}
                for nombre in columnas:
                    if nombre == "DIA":
                        fila[nombre] = d
                    elif nombre == "HORA":
                        fila[nombre] = hora
                    elif nombre == "COLUMNA_EXTRA_NO_USADA":
                        fila[nombre] = "valor ignorado"
                    else:
                        # Cada jornada inicia en cero (exigencia de calidad del frío)
                        # y crece a una tasa ligeramente distinta según la fecha.
                        fila[nombre] = float(indice * (200 + (d.day - 8) * 40))
                pagina.append([fila[c] for c in columnas])
    extra = libro.create_sheet("HOJA_EXTRA_NO_USADA")
    extra.append(["COLUMNA ARBITRARIA"])
    extra.append([123])
    libro.save(ruta)


def comprobar_error(ruta: Path, *, palabra: str) -> None:
    try:
        validar_entrada_excel(ruta)
    except ValueError as error:
        mensaje = str(error)
        if palabra not in mensaje:
            raise AssertionError(f"Error sin detalle esperado ({palabra!r}): {mensaje}") from error
        print("OK: error descriptivo:", mensaje)
    else:
        raise AssertionError(f"El Excel defectuoso fue aceptado: {ruta.name}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pruebas_fase4_") as temporal:
        carpeta = Path(temporal)
        extra = carpeta / "extra_y_columnas_reordenadas.xlsx"
        sin_columna = carpeta / "sin_columna.xlsx"
        sin_hoja = carpeta / "sin_hoja.xlsx"
        crear_excel(extra)
        crear_excel(sin_columna, sin_columna=("Consolidado EE", "Servicios (Kw)"))
        crear_excel(sin_hoja, sin_hoja="Consolidado Aire")

        print("\nPRUEBA 1: hojas extra y columnas extra/reordenadas")
        hojas = validar_entrada_excel(extra)
        assert len(hojas) == len(COLUMNAS_REQUERIDAS)
        print("OK: Excel aceptado por nombre de hoja y columna")

        print("\nPRUEBA 2: columna obligatoria ausente")
        comprobar_error(sin_columna, palabra="Servicios (Kw)")

        print("\nPRUEBA 3: hoja obligatoria ausente")
        comprobar_error(sin_hoja, palabra="Consolidado Aire")

        print("\nPRUEBA 4: versión de modelo inexistente")
        try:
            load_model("v999.0.0")
        except ValueError as error:
            assert "v999.0.0" in str(error)
            print("OK: error descriptivo:", error)
        else:
            raise AssertionError("Se aceptó una versión de modelo inexistente")

        print("\nPRUEBA 5: predicción con columnas reordenadas y extras")
        salida = predict_consumption(extra, version="latest")
        assert salida.columns.tolist() == ["fecha", "prediccion_frio_kw"]
        assert len(salida) == 2
        assert salida.fecha.is_unique
        assert np.isfinite(salida.prediccion_frio_kw.to_numpy()).all()
        print("OK: predicciones sintéticas diarias:", len(salida))

    print("\nPRUEBAS DE ROBUSTEZ: 5/5 CORRECTAS")
    print("Archivos temporales de prueba eliminados.")
    print("No se calcularon métricas de Test ni se modificaron los Excel originales.")


if __name__ == "__main__":
    main()
