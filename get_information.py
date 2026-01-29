import logging 
from pathlib import Path 
import pandas as pd 
import settings

settings.log_config()
logger = logging.getLogger(__name__)


file_list = ["EVOLUCION.txt", "TELEFONOS.txt"]

def get_data():
    """Obtención de información de la carpeta data dónde se alojan los tres archivos, 
    separador diferente en el archivo de pagos."""
    try:

        fisrt_route = str(Path.cwd()) + "/data/"
        df_pg = pd.read_csv(fisrt_route + "PAGOS.txt", sep='|',index_col=None, header=0, encoding='ISO-8859-1')
        df_list = [pd.read_csv(fisrt_route + filename, sep='\t', index_col=None, header=0, encoding='ISO-8859-1') for filename in file_list]
        logger.debug("Carga exitosa de los archivos.")
        df_ev = df_list[0]
        df_tel = df_list[1]

        return df_pg, df_ev, df_tel
        
    except Exception as err:
        logger.error(f"Error en la lectura del archivo {err}")

