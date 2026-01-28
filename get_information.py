import logging 
from pathlib import Path 
import pandas as pd 
import settings

settings.log_config()
logger = logging.getLogger(__name__)


file_list = ["EVOLUCION.txt", "TELEFONOS.txt"]

def get_data():
    try:

        fisrt_route = str(Path.cwd()) + "/data/"
        df_list_1 = pd.read_csv(fisrt_route + "PAGOS.txt", sep='|',index_col=None, header=0, encoding='ISO-8859-1')
        df_list = [pd.read_csv(fisrt_route + filename, sep='\t', index_col=None, header=0, encoding='ISO-8859-1') for filename in file_list]
        logger.info("Carga exitosa de los archivos.")
        print(df_list_1)
        print(df_list)
    except Exception as err:
        logger.error(f"Error en la lectura del archivo {err}")

