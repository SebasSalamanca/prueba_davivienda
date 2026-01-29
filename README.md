# Modelo de Predicción de Recaudo de Créditos

Proyecto de machine learning para predecir el monto de recaudo que los clientes pagarán sobre sus créditos, utilizando LightGBM como algoritmo principal.

## 🎯 Objetivo

Desarrollar un modelo predictivo que responda a la pregunta: **¿Cuánto puede llegar a pagar el cliente?** basándose en su saldo capital y otras características relevantes del portafolio de créditos.

## 📋 Descripción General

Este proyecto implementa un pipeline completo de ETL (Extract, Transform, Load) seguido de un modelo de regresión utilizando LightGBM para estimar los valores de pago de clientes. El sistema procesa datos de créditos, aplica reglas de negocio y entrena un modelo predictivo robusto.

## 🗂️ Estructura del Proyecto

```
├── main.py                    # Script principal que orquesta todo el proceso
├── settings.py                # Configuración del sistema y logging
├── get_information.py         # Extracción y carga de datos
├── transform_data.py          # Limpieza y transformación de datos
├── bussiness_rules.py        # Aplicación de reglas de negocio
├── modelo_recaudo.py          # Modelo de machine learning con LightGBM
├── etl.log                   # Bitácora de ejecución del proceso
├── modelo_recaudo.pkl        # Modelo entrenado serializado
└── RESPUESTAS_PUNTO_4.txt    #**IMPORTANTE: Contiene respuestas a preguntas finales**
```

## 🖼️ Visualizaciones Generadas

El modelo genera **tres imágenes fundamentales** para el análisis e interpretación de los resultados:

### 1. `predicciones_vs_reales.png`
- **Propósito**: Visualizar la precisión del modelo comparando valores predichos vs reales
- **Importancia**: Permite identificar si el modelo subestima o sobreestima los pagos
- **Interpretación**: Los puntos cercanos a la línea diagonal indican predicciones precisas

### 2. `importancia_caracteristicas.png`
- **Propósito**: Mostrar las variables más influyentes en las predicciones del modelo
- **Importancia**: **Fundamental** para entender qué factores determinan los pagos
- **Insights clave**: 
  - `PAGO_MINIMO` es la característica más importante (860,625 puntos de importancia)
  - `SALDO_CAPITAL_MES` es el segundo factor más relevante (23,411 puntos)
  - Variables de contacto telefónico también tienen peso significativo

### 3. `distribucion_ratio_pago.png`
- **Propósito**: Analizar la distribución del ratio de pago (pago/saldo capital)
- **Importancia**: Ayuda a entender el comportamiento de pago de los clientes
- **Utilidad**: Identificar patrones de pago y segmentos de clientes

## 📊 Métricas del Modelo

El modelo muestra un rendimiento moderado con las siguientes métricas principales:
- **MAE**: 22,836,438.63 (escala original)
- **RMSE**: 36,249,696.74 
- **R²**: 0.4781 (explica ~48% de la variabilidad)
- **MAPE**: 167.92% (error porcentual medio absoluto)

## 🚀 Ejecución

Para ejecutar el proyecto completo:

```bash
python main.py
```

O ejecutar solo el modelo:
```bash
python modelo_recaudo.py
```

## 📝 Proceso ETL

1. **Extracción**: Carga de archivos de datos del portafolio de créditos
2. **Transformación**: Limpieza de datos, manejo de nulos y estructuración
3. **Reglas de Negocio**: Aplicación de lógica específica del dominio financiero
4. **Exportación**: Generación de datos procesados para el modelo

## 🔧 Características del Modelo

- **Algoritmo**: LightGBM Regressor
- **Transformación**: Logarítmica de variable objetivo (sesgo = 1.75)
- **Preprocesamiento**: StandardScaler para variables numéricas
- **Validación**: División 80/20 para entrenamiento/prueba
- **Dataset**: 65,203 registros con 21 características numéricas

## ⚠️ Archivo Importante

**`RESPUESTAS_PUNTO_4.txt`** - Este archivo contiene las respuestas detalladas a las preguntas finales del proyecto. 

## 📈 Variables Más Influyentes

1. **PAGO_MINIMO**: Monto mínimo de pago (más importante)
2. **SALDO_CAPITAL_MES**: Saldo capital actual
3. **TELEFONO_2/TELEFONO_1**: Información de contacto
4. **CUENTA**: Identificador de cuenta
5. **SALDO_TOTAL_CLIENTE**: Deuda total del cliente

## 🛠️ Requisitos

- Python 3.12+
- pandas, numpy, scikit-learn
- lightgbm
- matplotlib, seaborn
- logging

## 📊 Generación de Resultados

El proceso genera automáticamente:
- Modelo serializado (`modelo_recaudo.pkl`)
- Tres visualizaciones clave para análisis
- Bitácora detallada de ejecución (`etl.log`)
- Métricas de evaluación del modelo

---

**Nota**: Para una comprensión completa del análisis, resultados y respuestas a las preguntas específicas del proyecto, consulte el archivo `RESPUESTAS_PUNTO_4.txt`.