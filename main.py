import settings
import get_information
import transform_data
import bussiness_rules
import logging
import modelo_recaudo


"""Descripción del ejercicio: Se realiza la carga de información de las fuentes: 
EVOLUCIÓN, PAGOS, y TELEFÓNOS con la finalidad de generar un archivo final
denominado df_evolucion_enriquecida el cual será usado dentro de dos modelos
1. Modelo de posibilidad de pago 2.Modelo de recaudo. Con el objetivo
de responder a la pregunta: ¿Cuanto puede llegar a pagar un cliente? """

# Configuración del sistema de logging
settings.log_config()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        # Paso 1: Extraer información
        logger.info("Iniciando proceso de carga de datos...")
        result = get_information.get_data()
        
        # Verificar que la carga fue exitosa
        if not result:
            logger.error("Error en la carga de datos. El proceso no puede continuar.")
            exit(1)
            
        df1, df2, df3 = result
        
        # Paso 2: Transformar datos
        logger.info("Limpiando y transformando datos...")
        df1, df2, df3 = transform_data.clean_info(df1, df2, df3)
        
        # Paso 3: Aplicar reglas de negocio
        logger.info("Aplicando reglas de negocio...")
        df = bussiness_rules.bussines_rule(df1, df2, df3)
        
        # Paso 4: Exportar datos procesados
        logger.info("Exportando datos procesados...")
        bussiness_rules.export_data(df)
    except Exception as e:
        logger.error(f"Error en el proceso ETL: {e}")
        logger.error("El proceso ETL ha fallado. No se puede continuar con el modelo.")
        exit(1)
    
    # Paso 5: Ejecutar el modelo de recaudo
    logger.info("Iniciando entrenamiento del modelo de recaudo...")
    try:
        # Crear instancia del modelo
        # Usamos el archivo generado en el paso anterior
        # El modelo leerá el archivo 'df_evolucion_enriquecida' que se generó con bussiness_rules.export_data(df)
        modelo = modelo_recaudo.ModeloRecaudo()
        
        # Ejecutar flujo completo del modelo
        resultados = modelo.ejecutar_flujo_completo()
        
        if resultados:
            metricas = resultados.get('metricas', {})
            logger.info("=== Resultados del Modelo de Recaudo ===")
            for metrica, valor in metricas.items():
                logger.info(f"{metrica}: {valor:.4f}")
            
            logger.info("Modelo de recaudo entrenado y guardado exitosamente.")
        else:
            logger.warning("No se obtuvieron resultados del modelo de recaudo.")
    except Exception as e:
        logger.error(f"Error al entrenar el modelo de recaudo: {e}")
        logger.info("Se completó el procesamiento de datos, pero hubo un error en el modelo.")
    