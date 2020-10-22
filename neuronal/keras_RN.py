from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.utils import to_categorical
import numpy as np

def precios( dataset, columns ):
    dataset = dataset.fillna(0.0)                                                                             # eemplazando valores vacios por ceros
    for column in columns:                                                                                    # Para cada nombre del conjunto de columnas
        C = []
        for value in dataset[column]:                                                                         # Para cada valor de cada columna
            try:
                C.append( float("".join((str(value).split("$")[-1]).split(","))) )                            # formatea el precio a flotante
            except:
                pass
        dataset[column] = pd.DataFrame(C)                                                                     # reemplaza la columna por los valores tipo flotante
    return dataset


def Keras_NN( dataset, sub_data):
    clase = []
    col = 'Nivel socioeconomico'
    C = pd.unique(dataset[col])
    for i in list(dataset[col]):
        if (i == C[0]):
            clase.append('0')
        elif (i == C[1]):
            clase.append('1')
        elif (i == C[2]):
            clase.append('2')
        elif (i == C[3]):
            clase.append('3')

    train_labels = to_categorical(clase)
    X = sub_data.iloc[:, :]
    y = train_labels

    model = Sequential()
    model.add(Dense(20, input_dim=15, activation='relu'))
    model.add(Dense(15, input_dim=15, activation='relu'))
    model.add(Dense(len(C), activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=15, batch_size=300)
    _, accuracy = model.evaluate(X, y)

    predictions = model.predict_classes(X)
    a = 0
    for i in range(len(sub_data)):
        pos_exp = np.where(y[i] == 1)[0]
        pos_prd = predictions[i]
        if( pos_prd == pos_exp ):
            a += 1
    dataset['Clase keras'] = predictions
    return dataset


dataset = pd.read_csv('../Escuelas_para_publicitar.csv')
costos_cols = dataset.iloc[:, dataset.columns.str.contains('Costo')]
sub_data = precios(costos_cols.fillna(0.0), costos_cols.columns)
new_dataset = Keras_NN(dataset, sub_data)
new_classes = pd.unique(new_dataset['Clase keras'])
while(len(new_classes) != 4):
    new_dataset = Keras_NN(dataset, sub_data)
    new_classes = pd.unique(new_dataset['Clase keras'])

dataset = dataset[['Nombre de Colegio', 'Calle', 'Colonia', 'Ciudades', 'Estados', 'Número Exterior',
                   'Correo electrónico', 'Codigo_Postal', 'coordenadas', 'Nivel socioeconomico',
                   'Clase AMAI', 'Clase keras']]                                                              # Eligiendo columnas de salida
dataset.to_csv('Escuelas_para_publicitar.csv', index=False)
