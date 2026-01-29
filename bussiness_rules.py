import pandas as pd
import numpy as np
import logging
import settings

settings.log_config()
logger = logging.getLogger(__name__)

def bussines_rule(df_pg, df_ev, df_tel):
    """Cruce de información de acuerdo a reglas de negocio."""

    #Regla asociada a cruces por cuenta.
    df_pg = df_pg.loc[df_pg["ESTADO_PAGO"] == "APROBADO", :]
    df_pg_ev = pd.merge(df_pg, df_ev, left_on='CUENTA', right_on='OBLIGACION', how='inner')

    resultados = []
    #Regla asociada a numeros de telefóno. 
    for id, grupo in df_tel.groupby('IDENTIFICACION'):
        celulares = []
        fijos = []
        for telefono in grupo['TELEFONO_1']:
            # Limpiar espacios
            telefono = str(telefono).strip()
            
            # REGLA 2: Validar celulares (10 dígitos, empieza con 7 seguido de 0-5)
            if len(telefono) == 10 and telefono.startswith('7') and telefono[1] in '012345':
                celulares.append(telefono)
            
            # REGLA 3: Validar fijos (7 dígitos, empieza con 1-5)
            elif len(telefono) == 7 and telefono[0] in '12345':
                fijos.append(telefono)
        
        # REGLA 1: Ordenar (celulares primero, luego fijos)
        telefonos_ordenados = celulares + fijos
        fila = {'IDENTIFICACION': id}
        
        for i, tel in enumerate(telefonos_ordenados, 1):
            fila[f'TELEFONO_{i}'] = tel
        
        resultados.append(fila)
    
    df_tel2 = pd.DataFrame(resultados)

    df_union = df_pg_ev.merge(df_tel2, on="IDENTIFICACION", how='inner')
    
    #Regla asociada a creación de columnas:
    count_customer_products = df_union.groupby("IDENTIFICACION")["PRODUCTO"].count()
    df_union = df_union.merge(count_customer_products, on="IDENTIFICACION", how='left')
    df_union = df_union.rename(columns={'PRODUCTO_x': 'PRODUCTO'})
    df_union = df_union.rename(columns={'PRODUCTO_y': 'CANTIDAD_PRODUCTOS'})
    
    #Creación de columna TIPO_CLIENTE
    df_union['TIPO_CLIENTE'] = np.where(df_union['CANTIDAD_PRODUCTOS'] > 1, 'MULTIPRODUCTO', 'MONOPRODUCTO')
    #Creación de columna ESTADO_ORIGEN
    df_union['ESTADO_ORIGEN'] = np.where(df_union['PRODUCTO'] == "CON_ACUERDO", 'CON_ACUERDO', 'SIN_ACUERDO')

    #Creación de columna SALDO_TOTAL_CLIENTE
    balance_customer = df_union.groupby("IDENTIFICACION")["SALDO_CAPITAL_MES"].sum()
    df_union = df_union.merge(balance_customer, on="IDENTIFICACION", how='left')
    df_union = df_union.rename(columns={'SALDO_CAPITAL_MES_x': 'SALDO_CAPITAL_MES'})
    df_union = df_union.rename(columns={'SALDO_CAPITAL_MES_y': 'SALDO_TOTAL_CLIENTE'})

    #Creación de columna SALDO_TOTAL_CLIENTE
    conditions = [
    df_union['DIAS_MORA'] == 0,
    df_union['DIAS_MORA'] < 30,
    df_union['DIAS_MORA'] < 60,
    df_union['DIAS_MORA'] < 90,
    df_union['DIAS_MORA'] < 120,
    df_union['DIAS_MORA'] < 180,
    df_union['DIAS_MORA'] < 360,
    df_union['DIAS_MORA'] < 540,
    df_union['DIAS_MORA'] >= 540
    ]   

    valores = [
        'AL DIA',
        'MENOS 30',
        'MENOS 60',
        'MENOS 90',
        'MENOS 120',
        'MENOS 180',
        'MENOS 360',
        'MENOS 540',
        'MAS DE 540'
    ]

    df_union['RANGO_DIAS_MORA'] = np.select(conditions, valores, default='SIN CLASIFICAR')

    #Creación de columna RANGO_MORA_CLIENTE
    #Usamos la función transform para mantener el tamaño del dataframe
    df_union['RANGO_MORA_CLIENTE'] = df_union.groupby('IDENTIFICACION')['DIAS_MORA'].transform('max')
     

    # Comparar PAGOS vs PAGO_MINIMO
    # Creación de columna CUMPLE_PAGO_MINIMO
    conditions_2 = [
        df_union['PAGOS'] >= df_union['PAGO_MINIMO'],  
        df_union['PAGOS'] >= (df_union['PAGO_MINIMO'] * 0.7),  
        df_union['PAGOS'] < (df_union['PAGO_MINIMO'] * 0.7)  
    ]

    valores = ['CUMPLE_TOTAL', 'CUMPLE_PARCIAL', 'NO_CUMPLE']
    df_union['CUMPLE_PAGO_MINIMO'] = np.select(conditions_2, valores, default='SIN_CLASIFICAR')

    logger.debug("Finalización de aplicación de reglas de negocio.")

    return df_union

def export_data(df):

    df.to_csv('df_evolucion_enriquecida.txt', sep='|', index=False)
    
    
    