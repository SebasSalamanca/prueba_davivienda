import logging

def log_config():
    """configuración de logger para el registro de la ejecución"""
    logging.basicConfig(
        level = logging.DEBUG, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        #filemode = 'w',
        #filename = 'etl.log'
    )