import settings
import get_information
import transform_data
import bussiness_rules


"""Descripción del ejercicio: Se realiza la carga de información de las fuentes: 
EVOLUCIÓN, PAGOS, y TELEFÓNOS con la finalidad de generar un archivo final
denominado df_evolucion_enriquecida el cual será usado dentro de dos modelos
1. Modelo de posibilidad de pago 2.Modelo de recaudo. Con el objetivo
de responder a la pregunta: ¿Cuanto puede llegar a pagar un cliente? """

if __name__ == '__main__':
    df1, df2, df3 = get_information.get_data()
    df1, df2, df3 = transform_data.clean_info(df1, df2, df3)
    bussiness_rules.bussines_rule(df1, df2, df3)

    