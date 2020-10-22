import pandas as pd
import numpy as np
from sklearn import svm
from random import choice as ch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from sklearn import svm

def precios(dataset, columns):
    dataset = dataset.fillna(0.0)                                                                             # eemplazando valores vacios por ceros
    for column in columns:
        dataset[column] = [float(''.join((str(i).split('$')[-1]).split(','))) for i in dataset[column]]
    return dataset


def regresion(Firmados, Posibles, columnas):
    X = pd.concat([Firmados[columnas], Posibles[columnas]], axis=0)
    Y = [1 for _ in range(len(Firmados))] + [0 for _ in range(len(Posibles))]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    escalar = StandardScaler()

    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)

    algoritmo = LogisticRegression()

    algoritmo.fit(X_train, y_train)

    y_pred = algoritmo.predict(X_test)

    print('------------------------ Resultados de regresión lineal ---------------------------------')

    print(classification_report(y_test, y_pred))

    matriz = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(matriz)

    exactitud = accuracy_score(y_test, y_pred)
    print('Exactitud del modelo:')
    print(exactitud)

    return algoritmo


def KNN(Firmados, Posibles, columnas):
    X = pd.concat([Firmados[columnas], Posibles[columnas]], axis=0)
    Y = [1 for _ in range(len(Firmados))] + [0 for _ in range(len(Posibles))]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    K = 3

    print('--------------------------------- Resultados de KNN ---------------------------------')
    knn = KNeighborsClassifier(K)
    knn.fit(X_train, y_train)

    print('Precision de KNN en el conjunto de entrenamiento: {:.2f}'.format(knn.score(X_train, y_train)))
    print('Precision de KNN en el conjunto de prueba: {:.2f}'.format(knn.score(X_test, y_test)))

    pred = knn.predict(X_test)

    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))

    pred_f = knn.predict(Posibles[columnas])
    Posibles['clase'] = pred_f
    data = Posibles.drop_duplicates()
    data = data[data['clase'] == 1 ]
    print('Escuelas obtenidas con knn: ', len(data), 'de', len(Posibles), 'iniciales')
    return knn


def SVM(Firmados, columnas):
    firmados = precios(Firmados, columnas)
    firmados = firmados[columnas]
    s = len(firmados)
    file = firmados.sample(n=int(s*0.7))

    normal = file.iloc[:20, :].fillna(0.0).reset_index()
    anormal = file.iloc[20:, :].fillna(0.0).reset_index()
    normal['Clase'] = 0
    anormal['Clase'] = 1

    p = int(len(normal)*0.7)
    train = normal.loc[:p, :]
    train = train.drop('Clase', 1)

    Y1 = normal.loc[p:, 'Clase']
    Y2 = anormal['Clase']
    Y_test = Y1.append(Y2)

    X_test1 = normal.loc[:, :].drop('Clase', 1)
    X_test2 = anormal.drop('Clase', 1)
    X_test = X_test1.append(X_test2)

    one = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.01)
    one.fit(train)

    prediction = one.predict(X_test.fillna(0.0))
    print('--------------------------------- Resultados de SVM ---------------------------------')
    unique, count = np.unique(prediction, return_counts=True)
    print(np.asarray((unique, count)).T)

    Y_test = Y_test.to_frame()
    Y_test = Y_test.reset_index()
    prediction = pd.DataFrame(prediction)
    prediction = prediction.rename(columns={0: 'predictions'})

    TP = FN = FP = TN = 0
    for j in range(len(Y_test)):
        if Y_test['Clase'][j] == 0 and prediction['predictions'][j] == 1:
            TP = TP+1
        elif Y_test['Clase'][j] == 0 and prediction['predictions'][j] == -1:
            FN = FN+1
        elif Y_test['Clase'][j] == 1 and prediction['predictions'][j] == 1:
            FP = FP+1
        else:
            TN = TN + 1
    print(TP,  FN,  FP,  TN)

    accuracy = (TP + TN)/(TP + FN + FP + TN)
    print('Precision: ', accuracy)

    try:
        sens = TP / (TP + FN)
        print('Sensibilidad: ', sens)
    except:
        pass

    spec = TN / (TN + FP)
    #print('especificidad: ', spec)
    return one


def execute():
    V = pd.read_csv('Firmados.csv')
    F = pd.read_csv('Descartados.csv')

# ------------------------------------------ ELIGIENDO COLUMNAS PARA CLASIFICADORES --------------------------
    Costos_Firmados = V.iloc[:, V.columns.str.contains('Costo')]
    Costos_Descartados = F.iloc[:, F.columns.str.contains('Costo')]
    col1 = list(Costos_Firmados.columns)
    col2 = list(Costos_Descartados.columns)
    inter = sorted(set(col1) & set(col2))[:-1]

    V = precios(V, inter)
    F = precios(F, inter)

    regresion(V,F, inter)
    KNN(V,F, inter)
    SVM(V, inter)
    return 0


if __name__=='__main__':
    execute()