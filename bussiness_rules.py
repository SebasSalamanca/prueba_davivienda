import pandas as pd
import logging
import settings

settings.log_config()
logger = logging.getLogger(__name__)

def bussines_rule(df_pg, df_ev, df_tel):
    """Cruce de información de acuerdo a reglas de negocio."""
    print(df_pg.info())
    print(df_ev.info())
    print(df_tel.info())
    
    df_pg = df_pg.loc[df_pg["ESTADO_PAGO"] == "APROBADO", :]
    df_inner = pd.merge(df_pg, df_ev, left_on='CUENTA', right_on='OBLIGACION', how='inner')
    print(df_inner)

    resultados = []

    for id, grupo in df_tel.groupby('IDENTIFICACION'):
        celulares = []
        fijos = []
        for telefono in grupo['TELEFONO_1']:
            # Limpiar espacios
            telefono = str(telefono).strip()
            
            # REGLA ii: Validar celulares (10 dígitos, empieza con 7 seguido de 0-5)
            if len(telefono) == 10 and telefono.startswith('7') and telefono[1] in '012345':
                celulares.append(telefono)
            
            # REGLA iii: Validar fijos (7 dígitos, empieza con 1-5)
            elif len(telefono) == 7 and telefono[0] in '12345':
                fijos.append(telefono)
        
        # REGLA i: Ordenar (celulares primero, luego fijos)
        telefonos_ordenados = celulares + fijos

        fila = {'IDENTIFICACION': id}
        
        
        for i, tel in enumerate(telefonos_ordenados, 1):
            fila[f'TELEFONO_{i}'] = tel
        
        resultados.append(fila)
    
    # Convertir a DataFrame
    resultado_df = pd.DataFrame(resultados)
    print(resultado_df)
    