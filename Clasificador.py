from datetime import date
import zcrmsdk
import requests as rq
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from glob import glob
from random import uniform as rd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def Guardar(file, nombre):
    file.to_csv(nombre + '.csv', index=False)
    spath = './' + nombre
    nombres = sorted(glob(spath))
    cant = len(nombres)
    if (cant > 0):
        return file.to_csv('./tidy/' + nombre + '_' + str(cant + 1) + ".csv", index=False)
    else:
        return file.to_csv('./tidy/' + nombre + '_1.csv', index=False)


def Estatus(dataset):
    Total_P = dataset.sum(axis=1)
    q1, q2, q3 = list(Total_P.quantile([0.25, 0.50, 0.75]))

    nivel = []
    for precio in dataset.sum(axis=1):
        if( precio <= q1 ):
            nivel.append('Bajo')
        elif( precio > q1 and precio <= q2 ):
            nivel.append('Medio')
        elif( precio > q2 and precio <= q3 ):
            nivel.append('Medio Alto')
        elif( precio > q3 ):
            nivel.append('Alto')
    return nivel


def Clase( valor ):
    clase = 'N/A'
    if( valor > 150 ):
        clase = 'AB'
    elif (valor > 116 and valor <= 159):
        clase = 'C+'
    elif (valor > 97 and valor <= 115):
        clase = 'C'
    elif (valor > 89 and valor <= 96):
        clase = 'C-'
    elif (valor > 68 and valor <= 88):
        clase = 'D+'
    elif (valor > 50 and valor <= 67):
        clase = 'D'
    elif (valor <= 50):
        clase = 'E'
    return clase


def AMAI(dataset):
    columnas = dataset.columns
    x = dataset.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    Dataset = pd.DataFrame(x_scaled, columns=columnas)
    q1 = list(Dataset['promedioOcupacionVivienda'].quantile([0.25, 0.50, 0.75]))
    q2 = list(Dataset['automovilPosesion'].quantile([0.50]))
    q3 = list(Dataset['escolaridad'].quantile([0.17, 0.34, 0.51, 0.68, 0.85]))
    Etiquetas = []
    for ind, fila in Dataset.iterrows():
        V = rd(0,1)
        row = list(fila[columnas])
        Puntos = 0
# ----------------------- dormitorios --------------------
        if( V <= row[0] ):
            Puntos += 12
        else:
            if( V > row[1] ):
                Puntos += 6
            else:
                Puntos += 17
# ------------------------ ocupantes -----------------------
        if(V < q1[0]):
            Puntos += 15
        elif(V >= q1[0] and V < q1[1] ):
            Puntos += 31
        elif(V >= q1[1] and V < q1[2]):
            Puntos += 46
        elif(V >= q1[2]):
            Puntos += 61
# --------------------------- autos -------------------------
        if(V <= q2[0]):
            Puntos += 18
        elif(V > q2[0]):
            Puntos += 37
# -------------------------- internet -----------------------
        if(V <= row[4]):
            Puntos += 31
# ------------------------ escolaridad ----------------------
        if(V < q3[0] ):
            Puntos += 22
        elif(V >= q3[0] and V < q3[1]):
            Puntos += 31
        elif(V >= q3[1] and V < q3[2]):
            Puntos += 35
        elif(V >= q3[2] and V < q3[3]):
            Puntos += 43
        elif(V >= q3[3] and V < q3[4]):
            Puntos += 73
        elif(V >= q3[4]):
            Puntos += 101
# ------------------------------------------------------------------------------------
        Etiquetas.append(Clase(Puntos))
    return Etiquetas


def Compatibilidad(Colegios, indicadores):
    porcentaje = Colegios[indicadores].sum(axis=1)
    p_c = []
    nivel = (porcentaje*100)/70
    for valor in nivel:
        if( valor > 100 ):
            p_c.append(100)
        else:
            p_c.append(valor)
    return p_c


def precios(dataset, columns):
    dataset = dataset.fillna(0.0)                                                                             # eemplazando valores vacios por ceros
    for column in columns:
        dataset[column] = [float(''.join((str(i).split('$')[-1]).split(','))) for i in dataset[column]]
    return dataset


def KNN(Firmados, Descartados, columnas):
    X = pd.concat([Firmados[columnas], Descartados[columnas]], axis=0)
    Y = [1 for _ in range(len(Firmados))] + [0 for _ in range(len(Descartados))]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    K = 3

    knn = KNeighborsClassifier(K)

    knn.fit(X_train, y_train)
    print('Precision de KNN en el conjunto de entrenamiento: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Precision de KNN en el conjunto de prueba: {:.2f}'.format(knn.score(X_test, y_test)))
    pred = knn.predict(X_test)
    print(classification_report(y_test, pred))
    return knn


def filtro_columnas(Colegios):
    fechas = ['Fecha no califica', 'Fecha primera visita LH', 'Fecha siguiente campaña', 'Fecha descartado']

    to_dummies = ['Contrato Actual Secundaria', 'Plataformas Kinder',
               'Programas Kinder', 'Plataformas Primaria', 'Certificaciones Primaria', 'Programas Primaria',
               'Programas Secundaria', 'Plataformas Secundaria', 'Certificaciones Secundaria',
               'Razones por las que Califica siguiente campaña', 'Razones por las que se Descarta', 'País',
               'Primera campaña', 'Razones por las que no califica', 'Categoría del colegio']
    dummies = pd.get_dummies(Colegios[to_dummies])
    Colegios = pd.concat([Colegios, dummies], axis=1)

    Bol = ['Tiene Secundaria', 'Tiene Primaria', 'Tiene Kinder', 'Idioma Primaria Inglés',
           'Idioma Primaria Español', 'Idioma Secundaria Inglés', 'Idioma Secundaria Español',
           'Tiene Niveles Libres?', 'Uso de Libretas Kinder', 'Uso de Libretas Secundaria',
           'Uso de Libretas Primaria']
    Colegios[Bol] = Colegios[Bol].astype(int)

    CeroUno = ['Phone', 'Website', 'Description', 'Pertence a un grupo', 'Idioma Kinder Español',
               'Idioma Kinder Inglés', 'Maestro hace uso de dispositivo electrónico',
               'Alumno hace uso de dispositivo electrónico', 'Colegio IB', 'Nivel de inglés Kinder',
               'Nivel de inglés primaria', 'Nivel de inglés Secundaria']
    for col in CeroUno:
        N = Colegios[col] != 0
        Colegios[col] = N.astype(int)

    tipo_disp = ['Tipo de dispositivo Kinder',
                 'Tipo de dispositivo Primaria',
                 'Tipo de dispositivo Secundaria']

    matriculas = ['Matrícula Kinder',
                  'Matrícula Primaria',
                  'Matrícula Secundaria']

    materiales = ['Costo de Materiales Kinder',
                  'Costo de Materiales Primaria',
                  'Costo de Materiales Secundaria']

    colegiaturas = ['Costos Colegiatura Kinder',
                    'Costos Colegiatura Primaria',
                    'Costos Colegiatura Secundaria']

    inscripcion = ['Costo Inscripción Kinder',
                   'Costo Inscripción Primaria',
                   'Costo Inscripción Secundaria']

    contrato = ['Fecha vencimiento contrato Kinder',
                'Fecha Vencimiento Contrato Primaria',
                'Fecha Vencimiento Contrato Secundaria']

    instalaciones = 'Nivel de Instalaciones'
    dispo_maestro = 'Maestro hace uso de dispositivo electrónico'
    dueno_dispo = 'Propietario de dispositivo'
    uso_dispo = 'Alumno hace uso de dispositivo electrónico'
    estructura = 'Estructura Propietaria'

    Colegios[matriculas] = (Colegios[matriculas].fillna(0)/100).apply(np.int64)                                         # 1 punto por cada 100
    Colegios[materiales] = (precios(Colegios[materiales], materiales)/1000).apply(np.int64)                   # 1 punto por cada 1000
    Colegios[inscripcion] = (precios(Colegios[inscripcion], inscripcion)/1000).apply(np.int64)                # 1 punto por cada 1000
    Colegios[colegiaturas] = (precios(Colegios[colegiaturas], colegiaturas)/1000).apply(np.int64)             # 2 puntos por cada 1000
    Colegios[instalaciones] = Colegios[instalaciones].map({0: 0, 'Bajo': 1, 'Medio': 2, 'Alto': 3})           # 1 punto por cada valor del nivel de instalacion
    Colegios[dispo_maestro] = Colegios[dispo_maestro].map({0: 0, 'No': 1, 'Si': 2})
    Colegios[dueno_dispo] = Colegios[dueno_dispo].map({0: 0, 'Colegio': 1,  'Alumno': 1, 'Arrendamiento': 1})
    Colegios[uso_dispo] = Colegios[uso_dispo].map({0: 0, 'No': 0, 'Otro': 1, '4 a 1': 2,                      # 1 punto por cada usuario de ipad
                                                   '3 a 1': 3, '2 a 1': 4, '1 a 1': 5})
    Unico = 'Dueño único que opera'                                                                           # Valores de la columna
    Unico_NO = 'Dueño único que no opera'
    Acc_ind = 'Dueño único que no opera Accionistas independientes Accionistas que operan el negocio'
    unico = Colegios[Colegios[estructura] == Unico].index                                                     # indices de los valores
    unico_NO = Colegios[Colegios[estructura] == Unico_NO].index
    acc_ind = Colegios[Colegios[estructura] == Acc_ind].index
    otros = Colegios[(Colegios[estructura] != Unico) &
                     (Colegios[estructura] != Unico_NO) &
                     (Colegios[estructura] != Acc_ind)].index
    Colegios.loc[list(unico), estructura] = 4
    Colegios.loc[list(unico_NO), estructura] = 3
    Colegios.loc[list(acc_ind), estructura] = 2
    Colegios.loc[list(otros), estructura] = 1

    for grado in tipo_disp:
        ipad_ind = Colegios[Colegios[grado] == 'Ipad'].index                                                  # Escuelas que usan ipad
        no_ipad = Colegios[(Colegios[grado] != 'Ipad') & Colegios[grado] != 0].index                          # Escuelas que no usan ipad y si otro dispositivo
        Colegios.loc[list(ipad_ind), grado] = 2                                                               # 2 puntos si usan ipan
        Colegios.loc[list(no_ipad), grado] = 1                                                                # 1 punto si usan algun otro dispositivo

    for columna in contrato:
        Colegios[columna] = pd.to_datetime(Colegios[columna].fillna(0)).dt.date - date.today()                          # calculando la diferencia de dias
        dias = Colegios[columna].astype(str).str.split(' ', n=1, expand=True)                                 # Solo el valor numerico
        Colegios[columna] = dias[0].apply(np.int64)                                                           # Convirtiendo en entero
        optimo = Colegios[Colegios[columna] <= 0][columna].index
        req_min = Colegios[(Colegios[columna] > 0) & (Colegios[columna] <= 180)].index
        dificl = Colegios[(Colegios[columna] > 180) & (Colegios[columna] <= 1080)].index
        no_cand = Colegios[(Colegios[columna] > 1080)].index
        Colegios.loc[list(optimo), columna] = 4
        Colegios.loc[list(req_min), columna] = 3
        Colegios.loc[list(dificl), columna] = 2
        Colegios.loc[list(no_cand), columna] = 1

    c = list(dummies.columns)
    columnas = c + Bol + CeroUno + tipo_disp + matriculas + materiales + colegiaturas + inscripcion +\
               contrato + [instalaciones] + [dispo_maestro] + [dueno_dispo] + [uso_dispo] + [estructura]
    return (Colegios, columnas)


def clasificacion(Colegios):
    Colegios.fillna(0, inplace=True)
    Colegios_col, indicadores = filtro_columnas(Colegios)

    Firmados = Colegios_col[Colegios_col['Estatus Colegio'] == 'Colegio Aliado'].fillna(0)
    Descartados = Colegios_col[Colegios_col['Estatus Colegio'] == 'Descartado'].fillna(0)
    Entrada = Colegios_col[(Colegios_col['Estatus Colegio'] != 'Descartado') &
                       (Colegios_col['Estatus Colegio'] != 'Aliado')].fillna(0)

# ------------------------------------------- PROCESO PARA KNN -----------------------------------------------
    Algoritmo = KNN(Firmados, Descartados, indicadores)
    Clases_predic = Algoritmo.predict(Entrada[indicadores])
    Entrada['Clase'] = Clases_predic

    data = Entrada[Entrada['Clase'] == 1]
    print('Escuelas obtenidas con knn: ', len(data), 'de', len(Entrada), 'iniciales')
    Guardar(data, 'Colegios_para_publicitar')

# ----------------------------------- DEFINIENDO CLASE DE PORCENTAJE DE COMPATBILIDAD ------------------------
    porcentajes = Compatibilidad(Entrada, indicadores)
    Entrada['Porcentaje de compatibilidad'] = porcentajes

# ---------------------------------------- DEFINIENDO CLASE DE NIVEL SOCIOECONOMICA --------------------------
    indicador_costo = Colegios.filter(like='Costo')
    indicador_matricula = Colegios.filter(like='Matrícula')
    cols = list(indicador_costo.columns) + list(indicador_matricula.columns)
    inter = sorted(cols)

    nivel_escuela = Estatus(Entrada[inter])                                                                   # Determinando el nivel de escuela segun insumos
    Entrada['Nivel Socioeconómico'] = nivel_escuela

# ----------------------------------------- AGREGANDO COLUMNAS AMAI A NUEVAS ENTRADAS ------------------------
    inegi_file = pd.read_csv('socioeconomicos_manzana.csv')                                                   # Archivo inegi para ligar datos socioeconomicos de cada escuela medienta el código postal
    matriz = []
    AMAI_cols = ['dosDorm', 'tresCuartos', 'promedioOcupacionVivienda',
                 'automovilPosesion', 'internetPosesion', 'escolaridad']                                      # Columnas necesarias para clasificacion amai
    for indice, escuela in enumerate(Entrada['Colegio Name']):                                                # Expresion de calculo de media -->  (A and (B or C))
        print('Calculando Media para... ', escuela)
        media = [i*0 for i in range(len(AMAI_cols))]
        try:
            media = (inegi_file[(inegi_file['nombreEstado'] == Entrada['Estados'][indice]) &
                                ((inegi_file['nombreMunicipio'] == Entrada['Ciudades'][indice].split('-')[0]) |
                                 (inegi_file['nombreLocalidad'] == Entrada['Ciudades'][indice].split('-')[0])) &
                                (inegi_file['CP'] == Entrada['Código Postal'][indice])].iloc[:, AMAI_cols]).mean()    # Calculado medias poblacionales de la region a código postal
        except:
            pass
        matriz.append(media)
    matriz = np.matrix(matriz)
    Entrada[AMAI_cols] = pd.DataFrame(matriz)                                                                # Agregando las medias a su registro correspondiente
    amai_clases = AMAI(Entrada[AMAI_cols])                                                                   # Determinando nivel de escuela segun amai
    Entrada['Clase AMAI'] = amai_clases                                                                      # Agregando la columna de clases
    return Entrada


def Extraccion_Direccion(Direccion):                                                                          # Extrayendo las coordenadas desde la direccion
    URL = 'https://maps.googleapis.com/maps/api/geocode/json?'
    key = 'AIzaSyDuVBF_MREeMtiguH13nGFIjvo821P2AVw'  # Key predeterminada
    URLcompleta = URL + 'address=' + Direccion + '&key=' + key
    info_Json = rq.get(URLcompleta).json()                                                                    # optimizar peticion?
    codigo_postal = info_Json['results'][0]['address_components'][-1]['long_name']
    return codigo_postal


def API_Places(file):
    coord = []
    postales = []
    for index, elemento in enumerate(file['Colegio Name']):                                                   # Procesando cada escuela del conjunto
        print('procesando ...', elemento)
        try:
            colegio = '%20'.join(elemento.split(' '))                                                         # modificando cadena para la url
            ciudad = '%20'.join((file.loc[index, 'Ciudades'].split('-')[0]).split(' '))
            direccion = colegio + '%20' + ciudad
            coordenadas, postal = Extraccion_Direccion(direccion)
            postales.append(str(postal))
            coord.append(dict(coordenadas))
        except:
            try:
                colegio = tildes('%20'.join(elemento.split(' ')))
                Calle = tildes('%20'.join(file.iloc[index, 1].split(' ')))
                Colonia = tildes('%20'.join(file.iloc[index, 2].split(' ')))
                Direccion = colegio + '%20' + Calle + '%20' + Colonia
                coordenadas, postal = Extraccion_Direccion(Direccion)
                postales.append(int(postal))
                coord.append(dict(coordenadas))
            except:
                coord.append('N/A')
                postales.append('N/A')
    file['Codigo_Postal'] = postales
    file['coordenadas'] = coord
    return file


def maps_link(file):
    mapas = []
    url_map = 'https://www.google.com/maps/search/?api=1&query='
    file = file.fillna(' ')
    for index, elemento in enumerate(file['Colegio Name']):                                                   # Procesando cada escuela del conjunto
        colegio_map = elemento
        calle_map = file['Calle'][index]
        col_map = file['Colonia'][index]
        try:
            url = url_map + colegio_map + ' ' + calle_map
        except:
            url = url_map + colegio_map + ' ' + col_map
        try:
            mapas.append('%20'.join(url.split(' ')))
        except:
            mapas.append('')
    file['Google Maps Link'] = mapas
    return file


def filtro_colegios(Colegios):
    Colegios = Colegios[(Colegios['Colegio Name'].notnull()) &
                        (Colegios['Colegio ID'].str.contains('zcrm'))]                                        # Colegios que tengan ID y nombre
    return Colegios


def execute():
    Colegios = filtro_colegios(pd.read_csv('Colegio.csv'))

    #Colegios = maps_link(Colegios)                                                                            # Calculando links google maps
    #Colegios = API_Places(Colegios)                                                                           # Calculando códigos postales
    Colegios_clasificados = clasificacion(Colegios)                                                           # Calculando las clases

    Guardar(Colegios_clasificados, 'Colegios_to_crm')

if __name__=='__main__':
    execute()