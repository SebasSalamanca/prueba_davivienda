#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modelo de Recaudo con LightGBM Regressor

Este script implementa un modelo de regresión utilizando LightGBM para estimar
el valor total que un cliente pagará de acuerdo a su saldo capital y otras características.

El modelo está diseñado para responder a la pregunta: 
¿Cuánto puede llegar a pagar el cliente?

Autor: OpenCode
Fecha: Enero 2026
"""

# ----- Importación de bibliotecas -----
import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# ----- Configuración de logging -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ----- Definición de clases y funciones -----

class ModeloRecaudo:
    """
    Clase principal para el modelo de recaudo que estima cuánto puede pagar un cliente.
    
    Esta clase implementa todo el flujo de trabajo del modelo:
    - Carga de datos
    - Preprocesamiento
    - Entrenamiento
    - Evaluación
    - Predicción
    """
    
    def __init__(self, ruta_datos='df_evolucion_enriquecida.txt'):
        """
        Inicializa el modelo con la ruta de los datos.
        
        Args:
            ruta_datos (str): Ruta al archivo de datos
        """
        self.ruta_datos = ruta_datos
        self.modelo = None
        self.preprocessor = None
        self.cat_columns = None
        self.num_columns = None
        self.var_objetivo = 'PAGOS'
    
    def cargar_datos(self):
        """
        Carga los datos desde el archivo especificado en formato pipe-delimitado.
        
        Returns:
            pandas.DataFrame: Conjunto de datos cargado
        """
        logger.info(f"Cargando datos desde {self.ruta_datos}")
        try:
            # Los datos están separados por pipe (|)
            df = pd.read_csv(self.ruta_datos, sep='|')
            logger.info(f"Se han cargado {df.shape[0]} registros con {df.shape[1]} características")
            return df
        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise
    
    def analizar_datos(self, df):
        """
        Realiza un análisis exploratorio básico de los datos.
        
        Args:
            df (pandas.DataFrame): Conjunto de datos a analizar
            
        Returns:
            pandas.DataFrame: Conjunto de datos con posibles transformaciones
        """
        logger.info("Analizando el conjunto de datos")
        
        # Verificar valores nulos
        nulos = df.isnull().sum()
        if nulos.sum() > 0:
            logger.warning(f"Se encontraron {nulos.sum()} valores nulos")
            logger.info("Imputando valores nulos")
            # Estrategia simple: rellenar númericos con 0 y categóricos con 'DESCONOCIDO'
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna('DESCONOCIDO', inplace=True)
                else:
                    df[col].fillna(0, inplace=True)
        
        # Verificar si la variable objetivo tiene valores negativos o cero
        if (df[self.var_objetivo] <= 0).any():
            logger.warning("La variable objetivo contiene valores no positivos")
            # Filtrar registros con valores no positivos en la variable objetivo
            df = df[df[self.var_objetivo] > 0]
            logger.info(f"Quedan {df.shape[0]} registros después de filtrar valores no positivos")
            
        # Identificar columnas categóricas y numéricas
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols.remove(self.var_objetivo)  # Excluir la variable objetivo
        
        self.cat_columns = cat_cols
        self.num_columns = num_cols
        
        logger.info(f"Características categóricas: {len(cat_cols)}")
        logger.info(f"Características numéricas: {len(num_cols)}")
        
        # Verificar cardinalidad de variables categóricas
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > 1000:
                logger.warning(f"Alta cardinalidad en {col}: {n_unique} valores únicos")
                
        # Aplicar transformación logarítmica a la variable objetivo si está sesgada
        if df[self.var_objetivo].skew() > 1:
            logger.info(f"Variable objetivo sesgada (skew={df[self.var_objetivo].skew():.2f}), aplicando transformación logarítmica")
            df['PAGOS_LOG'] = np.log1p(df[self.var_objetivo])
            self.var_objetivo = 'PAGOS_LOG'
            self.log_transform = True
        else:
            self.log_transform = False
        
        return df
    
    def preparar_datos(self, df):
        """
        Prepara los datos para el entrenamiento del modelo.
        
        Args:
            df (pandas.DataFrame): Conjunto de datos a preparar
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info("Preparando datos para el entrenamiento")
        
        # Separar características y variable objetivo
        X = df.drop(columns=[self.var_objetivo])
        if self.log_transform and 'PAGOS' in X.columns:
            X = X.drop(columns=['PAGOS'])
        y = df[self.var_objetivo]
        
        # División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
        logger.info(f"Conjunto de prueba: {X_test.shape[0]} muestras")
        
        # Crear preprocesador para características categóricas y numéricas
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        numeric_transformer = StandardScaler()
        
        # Crear columntransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.num_columns),
                ('cat', categorical_transformer, self.cat_columns)
            ])
        
        self.preprocessor = preprocessor
        
        # Aplicar preprocesamiento
        logger.info("Aplicando preprocesamiento a los datos")
        X_train_prep = self.preprocessor.fit_transform(X_train)
        X_test_prep = self.preprocessor.transform(X_test)
        
        return X_train_prep, X_test_prep, y_train, y_test, X_train, X_test
    
    def entrenar_modelo(self, X_train, y_train):
        """
        Entrena el modelo LightGBM con los datos preparados.
        
        Args:
            X_train: Características de entrenamiento procesadas
            y_train: Variable objetivo de entrenamiento
            
        Returns:
            lightgbm.LGBMRegressor: Modelo entrenado
        """
        logger.info("Entrenando modelo LightGBM Regressor")
        
        # Crear conjunto de datos LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Parámetros iniciales
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Entrenar modelo con parámetros compatibles
        callbacks = []
        try:
            # Intentar crear callbacks de LightGBM
            callbacks = [
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        except AttributeError:
            logger.warning("Versión de LightGBM no soporta callbacks - usando parámetros directos")
            # Soporte para versiones anteriores
            params['early_stopping_rounds'] = 50
            params['verbose_eval'] = 100
            
        try:
            if callbacks:
                # Para versiones más recientes con callbacks
                self.modelo = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[train_data],
                    callbacks=callbacks
                )
            else:
                # Para versiones más antiguas
                self.modelo = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[train_data],
                    early_stopping_rounds=50,
                    verbose_eval=100
                )
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {e}")
            # Fallback: entrenar modelo con parámetros mínimos
            logger.info("Intentando entrenar con configuración mínima")
            self.modelo = lgb.train(
                params,
                train_data,
                num_boost_round=1000
            )
        
        logger.info(f"Modelo entrenado con {self.modelo.num_trees()} árboles")
        
        return self.modelo
    
    def evaluar_modelo(self, X_test, y_test, X_test_orig=None):
        """
        Evalúa el rendimiento del modelo en el conjunto de prueba.
        
        Args:
            X_test: Características de prueba procesadas
            y_test: Variable objetivo de prueba
            X_test_orig: Características originales sin procesar
            
        Returns:
            dict: Métricas de evaluación
        """
        logger.info("Evaluando el modelo")
        
        # Realizar predicciones
        y_pred = self.modelo.predict(X_test)
        
        if self.log_transform:
            # Revertir transformación logarítmica
            y_pred_original = np.expm1(y_pred)
            y_test_original = np.expm1(y_test)
            
            # Calcular métricas en escala original
            mae = mean_absolute_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            r2 = r2_score(y_test_original, y_pred_original)
            
            logger.info(f"MAE (escala original): {mae:.2f}")
            logger.info(f"RMSE (escala original): {rmse:.2f}")
            logger.info(f"R² (escala original): {r2:.4f}")
            
            # Calcular métricas en escala logarítmica
            mae_log = mean_absolute_error(y_test, y_pred)
            rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
            r2_log = r2_score(y_test, y_pred)
            
            logger.info(f"MAE (escala log): {mae_log:.4f}")
            logger.info(f"RMSE (escala log): {rmse_log:.4f}")
            logger.info(f"R² (escala log): {r2_log:.4f}")
            
            # Si tenemos los datos originales, calculamos el error porcentual
            if X_test_orig is not None:
                mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
                logger.info(f"MAPE: {mape:.2f}%")
                
                # Visualización de predicciones vs valores reales
                self._visualizar_predicciones(y_test_original, y_pred_original, X_test_orig)
                
            return {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAE_log': mae_log,
                'RMSE_log': rmse_log,
                'R2_log': r2_log
            }
        else:
            # Calcular métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"R²: {r2:.4f}")
            
            # Si tenemos los datos originales, calculamos el error porcentual
            if X_test_orig is not None:
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                logger.info(f"MAPE: {mape:.2f}%")
                
                # Visualización de predicciones vs valores reales
                self._visualizar_predicciones(y_test, y_pred, X_test_orig)
                
            return {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
    
    def _visualizar_predicciones(self, y_true, y_pred, X_test):
        """
        Visualiza las predicciones versus los valores reales.
        
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
            X_test: Características de prueba
        """
        # Crear dataframe con valores reales y predichos
        results_df = pd.DataFrame({
            'Real': y_true,
            'Predicho': y_pred
        })
        
        # Si hay datos de saldo capital disponibles, agregar al dataframe
        if 'SALDO_CAPITAL_MES' in X_test.columns:
            results_df['SALDO_CAPITAL'] = X_test['SALDO_CAPITAL_MES'].values
            
            # Calcular ratio de pago (cuánto del saldo se paga)
            results_df['Ratio_Real'] = results_df['Real'] / results_df['SALDO_CAPITAL']
            results_df['Ratio_Predicho'] = results_df['Predicho'] / results_df['SALDO_CAPITAL']
        
        # Guardar visualizaciones
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('Valor real de pago')
        plt.ylabel('Valor predicho de pago')
        plt.title('Predicciones vs Valores Reales')
        plt.savefig('predicciones_vs_reales.png')
        
        if 'SALDO_CAPITAL' in results_df.columns:
                try:
                    plt.figure(figsize=(10, 6))
                    # Usar un método más compatible para las visualizaciones
                    # Primero, asegurarse que results_df es realmente un DataFrame
                    if not isinstance(results_df, pd.DataFrame):
                        logger.warning("Convirtiendo results_df a DataFrame para visualización")
                        if hasattr(results_df, 'to_frame'):
                            results_df = results_df.to_frame()
                        else:
                            results_df = pd.DataFrame(results_df)
                    
                    # Verificar que las columnas existen
                    if 'Ratio_Real' in results_df.columns and 'Ratio_Predicho' in results_df.columns:
                        # Enfoque alternativo más seguro para visualizaciones
                        plt.hist(results_df['Ratio_Real'].values, bins=30, alpha=0.5, label='Real', density=True)
                        plt.hist(results_df['Ratio_Predicho'].values, bins=30, alpha=0.5, label='Predicho', density=True)
                        plt.xlabel('Ratio de Pago (Pago/Saldo)')
                        plt.title('Distribución de Ratio de Pago')
                        plt.legend()
                        plt.savefig('distribucion_ratio_pago.png')
                    else:
                        logger.warning("Columnas Ratio_Real o Ratio_Predicho no encontradas para visualización")
                except Exception as viz_error:
                    logger.error(f"Error al crear visualización de distribución: {viz_error}")
            
        logger.info("Se han generado visualizaciones de las predicciones")
    
    def analizar_caracteristicas(self):
        """
        Analiza la importancia de las características en el modelo.
        """
        if self.modelo is None:
            logger.error("El modelo no ha sido entrenado aún")
            return
        
        # Obtener importancia de características
        feature_importance = self.modelo.feature_importance(importance_type='gain')
        
        # Obtener nombres de características
        feature_names = []
        for name, _, cols in self.preprocessor.transformers_:
            if name == 'cat':
                # Para características categóricas, obtener nombres de OneHotEncoder
                encoder = self.preprocessor.named_transformers_['cat']
                cat_features = encoder.get_feature_names_out(cols)
                feature_names.extend(cat_features)
            else:
                # Para características numéricas, usar nombres originales
                feature_names.extend(cols)
        
        # Crear dataframe de importancia
        feature_imp = pd.DataFrame({
            'Feature': feature_names[:len(feature_importance)],
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        # Mostrar las 20 características más importantes
        top_features = feature_imp.head(20)
        logger.info("Top 20 características más importantes:")
        for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
            logger.info(f"{i+1}. {feature}: {importance:.2f}")
        
        # Visualizar importancia de características
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title('Top 20 Características Más Importantes')
        plt.tight_layout()
        plt.savefig('importancia_caracteristicas.png')
        logger.info("Se ha generado visualización de importancia de características")
        
        return feature_imp
    
    def guardar_modelo(self, ruta_modelo='modelo_recaudo.pkl'):
        """
        Guarda el modelo entrenado y el preprocesador en disco.
        
        Args:
            ruta_modelo (str): Ruta donde guardar el modelo
        """
        if self.modelo is None:
            logger.error("No hay modelo para guardar")
            return
        
        # Guardar modelo y preprocesador
        with open(ruta_modelo, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'preprocessor': self.preprocessor,
                'cat_columns': self.cat_columns,
                'num_columns': self.num_columns,
                'log_transform': self.log_transform
            }, f)
        
        logger.info(f"Modelo guardado en {ruta_modelo}")
    
    def cargar_modelo(self, ruta_modelo='modelo_recaudo.pkl'):
        """
        Carga un modelo previamente guardado.
        
        Args:
            ruta_modelo (str): Ruta del modelo guardado
        """
        if not os.path.exists(ruta_modelo):
            logger.error(f"No se encuentra el archivo {ruta_modelo}")
            return False
        
        try:
            # Cargar modelo
            with open(ruta_modelo, 'rb') as f:
                datos = pickle.load(f)
            
            self.modelo = datos['modelo']
            self.preprocessor = datos['preprocessor']
            self.cat_columns = datos['cat_columns']
            self.num_columns = datos['num_columns']
            self.log_transform = datos.get('log_transform', False)
            
            logger.info(f"Modelo cargado desde {ruta_modelo}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            return False
    
    def predecir(self, datos_nuevos):
        """
        Realiza predicciones para nuevos datos.
        
        Args:
            datos_nuevos (pandas.DataFrame): Datos para los que se quiere predecir
            
        Returns:
            numpy.ndarray: Predicciones
        """
        if self.modelo is None:
            logger.error("El modelo no ha sido cargado o entrenado")
            return None
        
        try:
            # Verificar columnas
            for col in self.cat_columns + self.num_columns:
                if col not in datos_nuevos.columns:
                    logger.warning(f"Columna {col} no encontrada en los datos nuevos")
            
            # Preprocesar datos
            X_prep = self.preprocessor.transform(datos_nuevos)
            
            # Realizar predicción
            predicciones = self.modelo.predict(X_prep)
            
            # Si se aplicó transformación logarítmica, revertirla
            if self.log_transform:
                predicciones = np.expm1(predicciones)
            
            logger.info(f"Se han generado {len(predicciones)} predicciones")
            return predicciones
        
        except Exception as e:
            logger.error(f"Error al realizar predicciones: {e}")
            return None
    
    def ejecutar_flujo_completo(self):
        """
        Ejecuta el flujo completo de entrenamiento del modelo.
        
        Returns:
            dict: Resultados del entrenamiento y evaluación
        """
        try:
            # Cargar datos
            df = self.cargar_datos()
            
            # Analizar datos
            df = self.analizar_datos(df)
            
            # Preparar datos
            X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = self.preparar_datos(df)
            
            # Entrenar modelo
            self.entrenar_modelo(X_train, y_train)
            
            # Evaluar modelo
            metricas = self.evaluar_modelo(X_test, y_test, X_test_orig)
            
            # Analizar características
            importancia = self.analizar_caracteristicas()
            
            # Guardar modelo
            self.guardar_modelo()
            
            return {
                'metricas': metricas,
                'importancia': importancia
            }
            
        except Exception as e:
            logger.error(f"Error en el flujo de trabajo: {e}")
            raise


def main():
    """
    Función principal que ejecuta el flujo completo.
    """
    # Configurar directorio de trabajo
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Crear instancia del modelo
    modelo = ModeloRecaudo()
    
    # Ejecutar flujo completo
    try:
        resultados = modelo.ejecutar_flujo_completo()
        logger.info("Ejecución completada exitosamente")
        return resultados
    except Exception as e:
        logger.error(f"Error en la ejecución: {e}")
        return None


if __name__ == '__main__':
    main()