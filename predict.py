import os
import joblib
import pandas as pd

# Rutas a los artefactos serializados en data/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "data", "modelo_random_forest_suaci.pkl")
RUTA_PREPROCESADOR = os.path.join(BASE_DIR, "data", "preprocesador_suaci.pkl")

# Umbral de negocio validado en la Celda 6
UMBRAL_DECISION = 0.40

_modelo = None
_preprocesador = None


def cargar_artefactos(ruta_modelo=RUTA_MODELO, ruta_preprocesador=RUTA_PREPROCESADOR):
    """Carga en memoria el modelo y el preprocesador exportados."""
    global _modelo, _preprocesador
    if _modelo is None or _preprocesador is None:
        if not os.path.exists(ruta_modelo) or not os.path.exists(ruta_preprocesador):
            raise FileNotFoundError(
                f"No se encontraron los artefactos en '{ruta_modelo}' o '{ruta_preprocesador}'. "
                "Verifica haber ejecutado la Celda 4 del notebook 02_modelado.ipynb."
            )
        _modelo = joblib.load(ruta_modelo)
        _preprocesador = joblib.load(ruta_preprocesador)
    return _modelo, _preprocesador


def predecir_reclamo(datos_reclamo: dict, umbral: float = UMBRAL_DECISION) -> dict:
    """
    Recibe un reclamo individual y retorna su predicción de resolución.
    
    Columnas requeridas según el preprocesador entrenado:
      - Categóricas: 'categoria', 'tipo', 'canal', 'comuna', 'dia_semana'
      - Numéricas:   'mes', 'hora', 'es_fin_de_semana'
    """
    modelo, preprocesador = cargar_artefactos()

    df_entrada = pd.DataFrame([datos_reclamo])

    # Normalización idéntica a la etapa de limpieza
    cols_texto = ['categoria', 'tipo', 'canal']
    for col in cols_texto:
        if col in df_entrada.columns:
            df_entrada[col] = df_entrada[col].fillna('DESCONOCIDO').astype(str).str.upper().str.strip()
        else:
            df_entrada[col] = 'DESCONOCIDO'

    if 'comuna' in df_entrada.columns:
        df_entrada['comuna'] = df_entrada['comuna'].fillna('-1').astype(str)
    else:
        df_entrada['comuna'] = '-1'

    if 'dia_semana' in df_entrada.columns:
        df_entrada['dia_semana'] = df_entrada['dia_semana'].astype(str)
    else:
        df_entrada['dia_semana'] = 'Monday'

    # Validar tipos numéricos
    df_entrada['mes'] = df_entrada.get('mes', 1).astype(int)
    df_entrada['hora'] = df_entrada.get('hora', 12).astype(int)
    df_entrada['es_fin_de_semana'] = df_entrada.get('es_fin_de_semana', 0).astype(int)

    # 1. Aplicar el One-Hot Encoder y passthrough numérico (genera 85 variables)
    X_prep = preprocesador.transform(df_entrada)

    # 2. Calcular probabilidades con el Random Forest
    prob_cerrado = float(modelo.predict_proba(X_prep)[0, 1])
    prob_no_resuelto = 1.0 - prob_cerrado

    # 3. Aplicar corte con el umbral operativo
    prediccion_binaria = 1 if prob_cerrado >= umbral else 0

    return {
        "prediccion_id": prediccion_binaria,
        "estado_predicho": "Cerrado" if prediccion_binaria == 1 else "No Resuelto / Demorado",
        "probabilidad_cierre": round(prob_cerrado * 100, 2),
        "probabilidad_no_resuelto": round(prob_no_resuelto * 100, 2),
        "alerta_demora": prediccion_binaria == 0,
        "umbral_utilizado": umbral
    }


if __name__ == "__main__":
    print("--- Test del Módulo de Inferencia SUACI ---")

    # Caso 1: Reclamo típico con alta tasa de cierre
    caso_exitoso = {
        "categoria": "HIGIENE",
        "tipo": "REPORTE",
        "canal": "BOTI",
        "comuna": "14",
        "dia_semana": "Monday",
        "mes": 3,
        "hora": 10,
        "es_fin_de_semana": 0
    }

    # Caso 2: Reclamo atípico / con riesgo operativo de traba
    caso_riesgo = {
        "categoria": "TRÁNSITO",
        "tipo": "QUEJA",
        "canal": "WEB",
        "comuna": "-1",
        "dia_semana": "Sunday",
        "mes": 11,
        "hora": 23,
        "es_fin_de_semana": 1
    }

    # Caso 3: Mi caso personalizado de prueba
    mi_propio_caso = {
        "categoria": "ALUMBRADO",
        "tipo": "SOLICITUD",
        "canal": "BOTI",
        "comuna": "3",
        "dia_semana": "Wednesday",
        "mes": 9,
        "hora": 14,
        "es_fin_de_semana": 0
    }

    print("\n[Predicción Caso 3 - Mi Caso Personalizado]:")
    print(predecir_reclamo(mi_propio_caso))

    print("\n[Predicción Caso 1 - Higiene / Palermo]:")
    print(predecir_reclamo(caso_exitoso))

    print("\n[Predicción Caso 2 - Tránsito / Sin Comuna]:")
    print(predecir_reclamo(caso_riesgo))