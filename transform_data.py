import re
import pandas as pd
import numpy as np
import logging
import settings


settings.log_config()
logger = logging.getLogger(__name__)

def clean_info(df_pg, df_ev, df_tel):
    """Función para limpieza de los datos en las tablas"""

    df_pg.drop_duplicates()
    df_ev.drop_duplicates()
    df_tel.drop_duplicates()
    
    #Estandarizar el nombre de las columnas mayúsculas y rellenar espacios con_
    df_ev.columns = df_ev.columns.str.upper().str.strip().str.replace(' ', '_')
    df_tel.columns = df_tel.columns.str.upper().str.strip().str.replace(' ', '_')
    
    
    def clean_pagos(df_pg):
        ##TABLA PAGOS 
        try:
            pd.options.display.float_format = '{:.2f}'.format
            df_pg['PAGOS'] = df_pg['PAGOS'].str.replace(',', '.').astype(float).round(2)
            logger.debug(f"Limpieza de la tabla PAGOS finalizada")
            return df_pg
        except Exception as err:
            logger.error(f"Error en la limpieza de la tabla PAGOS {err}")

    def clean_evolucion(df_ev):
        #TABLA EVOLUCIÓN
        try:
            #Column obligación
            df_ev["OBLIGACION"] = df_ev["OBLIGACION"].str.replace(r'CTA_', '', regex=True).astype(int)
            #Columna nombre
            df_ev['NOMBRE'] = df_ev['NOMBRE'].str.replace(r'[~*-]', '', regex=True) 

            def clean_name(column):
            # Dividir por el primer _
                divide_name = column.split('_', 1)
                first_part = divide_name[0]  
                second_part = divide_name[1]     
                clean_second = re.sub(r'[^A-Za-z0-9]', '', second_part)
                return f"{first_part}_{clean_second}"

            df_ev["NOMBRE"] = df_ev["NOMBRE"].apply(clean_name)

            #Columna producto
            df_ev["PRODUCTO"] = df_ev["PRODUCTO"].str.lower().str.replace(r'[^a-zA-Z0-9]', '', regex=True)
            df_ev["PRODUCTO"] = df_ev["PRODUCTO"].str.replace(r'crdito', 'credito_', regex=True)

            def clean_product(serie):
                "Limpieza de la columna producto"
                valor = str(serie).lower().strip()

                # Patrones con regex (orden importa!)
                if re.search(r'tarjeta|tc|visa|master', valor):
                    return 'TARJETA_CREDITO'
                elif re.search(r'vehicul|veh|vhp|moto', valor):
                    return 'VEHICULO'
                elif re.search(r'libran', valor):
                    return 'LIBRANZA'
                elif re.search(r'hipotec', valor):
                    return 'HIPOTECARIO'
                elif re.search(r'libreinv', valor):
                    return 'LIBRE_INVERSION'
                elif re.search(r'cxrotativo|rotativo', valor):
                    return 'CREDITO_ROTATIVO'
                elif re.search(r'sobregir|sobregro', valor):
                    return 'SOBREGIRO'
                elif re.search(r'adelanto|nomina|pension|sueldo', valor):
                    return 'ADELANTO_NOMINA'
                elif re.search(r'consumo|prestamo', valor):
                    return 'CONSUMO'
                elif re.search(r'acuerdo', valor):
                    return 'ACUERDO_PAGO'
                elif re.search(r'nano', valor):
                    return 'NANOCREDITO'
                elif re.search(r'colegio|colee', valor):
                    return 'COLEGIO'
                elif re.search(r'club', valor):
                    return 'CLUB'
                elif re.search(r'cxfijo', valor):
                    return 'CREDITO_FIJO'
                elif re.search(r'fng', valor):
                    return 'FNG'
                elif re.search(r'ffmm', valor):
                    return 'FFMM'
                elif re.search(r'insolven', valor):
                    return 'INSOLVENCIA'
                elif re.search(r'cartera', valor):
                    return 'CARTERA'
                elif re.search(r'retail', valor):
                    return 'RETAIL'
                elif re.search(r'comercio', valor):
                    return 'COMERCIO'
                elif re.search(r'marca', valor):
                    return 'MARCAS'
                elif re.search(r'mprivada|mcompartida|pension', valor):
                    return 'PENSIONES'
                elif re.search(r'generic', valor):
                    return 'GENERICO'
                elif re.search(r'normali', valor):
                    return 'NORMALIZACION'
                #else:
                    #return 'sin_clasificar'

            df_ev["PRODUCTO"] = df_ev["PRODUCTO"].apply(clean_product)

            #Columna SALDO_CAPITAL_MES y PAGO_MINIMO

            df_ev["SALDO_CAPITAL_MES"] = df_ev["SALDO_CAPITAL_MES"].str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True).astype(float)
            df_ev["PAGO_MINIMO"] = df_ev["PAGO_MINIMO"].str.strip().str.replace(r'[^a-zA-Z0-9]', '', regex=True).astype(float)
            logger.debug(f"Limpieza de la tabla EVOLUCION finalizada")
            return df_ev
        
        except Exception as err:
            logger.error(f"Error en la limpieza de la tabla evolución {err}")

    #TABLA TELEFÓNO
    def clean_phone(df_tel):
        try:
            df_tel["IDENTIFICACION"] = df_tel["IDENTIFICACION"].str.strip()
            df_tel["TELEFONO_1"] = df_tel["TELEFONO_1"].astype('Int64').astype(str)
            df_tel['LONGITUD'] = df_tel['TELEFONO_1'].str.len()
            #df_tel = df_tel.loc[df_tel['LONGITUD'].isin([7, 10]), :]
            df_tel = df_tel[df_tel['LONGITUD'].isin([7, 10])]
            logger.debug(f"Limpieza de la tabla TELEFONO finalizada")
            return df_tel

        except Exception as err:
            logger.error(f"Error en la limpieza de la tabla evolución {err}")

    df_pg = clean_pagos(df_pg)
    df_ev = clean_evolucion(df_ev)
    df_tel = clean_phone(df_tel)

    return df_pg, df_ev, df_tel
