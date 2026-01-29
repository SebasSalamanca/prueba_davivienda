import logging

def log_config():
    """configuración de logger para el registro de la ejecución"""
    logging.basicConfig(
        level = logging.DEBUG, 
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode = 'w',
        filename = 'etl.log'
    )
    
    # Ignorar logs de matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.WARNING)