from pybrain3.datasets            import ClassificationDataSet
from pybrain3.utilities           import percentError
from pybrain3.tools.shortcuts     import buildNetwork
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules   import SoftmaxLayer
from pybrain3.datasets            import SupervisedDataSet
from matplotlib.pyplot            import ion, ioff, figure, draw, contourf, clf, show, plot
from sklearn                      import preprocessing
from scipy                        import arange, meshgrid
import pandas as pd
import random as rd
import pickle
import numpy as np
import pybrain3


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


def perceptron(dataset, sub_data):
    clase = []
    C = pd.unique(dataset['Nivel socioeconomico'])
    for i in list(dataset['Nivel socioeconomico']):
        if( i == C[0] ):
            clase.append(0)
        elif(i == C[1]):
            clase.append(1)
        elif(i == C[2]):
            clase.append(2)
        elif(i == C[3]):
            clase.append(3)

    CDS = ClassificationDataSet(15, nb_classes=4)
    cont = 0
    for i in range(len(sub_data)):
        row = np.array(list(sub_data.iloc[cont, :]))
        CDS.addSample(row, clase[cont])
        cont += 1
    test, train = CDS.splitWithProportion(0.25)
    test._convertToOneOfMany(bounds=[0,1,2,3])
    train._convertToOneOfMany(bounds=[0,1,2,3])

    network = buildNetwork(train.indim, 4, train.outdim, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(network, dataset=train, momentum=0.1, verbose=True, weightdecay=0.01)

    trainer.trainEpochs(20)
    trnresult = percentError(trainer.testOnClassData(), train['class'])
    tstresult = percentError(trainer.testOnClassData(dataset=test), test['class'])

    '''
    file = open('Perceptron_pickle.xml', 'wb')
    pickle.dump(network, file)
    file.close()
    '''

    clases = trainer.testOnClassData(dataset=test)
    return clases


c = 0
b = pd.read_csv('Escuelas_para_publicitar.csv')
costos_cols = b.iloc[:, b.columns.str.contains('Costo')]
sub_data = precios(costos_cols.fillna(0.0), costos_cols.columns)
output = perceptron(b, sub_data)
vs = {'Bajo':0, 'Medio':1, 'Medio Alto':2, 'Alto':3}
for i, v in enumerate(output):
    v2 = vs[b['Nivel socioeconomico'].iloc[i]]
    if( v == v2 ):
        c += 1

print('\n porcentaje de clasificacion igual:', len(sub_data)/c)
