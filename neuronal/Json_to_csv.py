# encoding: utf-8
import json
import pandas as pd

Colegios = pd.DataFrame(columns = ['Código knotion','Nombre de Colegio', 'Calle', 'Número', 'Código Postal',
                                   'Ciudad', 'Estado', 'País'] )

with open('../data/school_data.json', 'r') as myfile:                                                                   # Leyendo archivo Json de escuelas firmadas
    data = myfile.read()
firmadas = json.loads(data)

size = len(firmadas)
Colegios['Código knotion'] = [ firmadas[i]['knotionCode'] for i in range(size) ]
Colegios['Nombre de Colegio'] = [ firmadas[i]['campusName'] for i in range(size) ]
Colegios['Calle'] = [ firmadas[i]['streetName'] for i in range(size) ]
Colegios['Número'] = [ firmadas[i]['exteriorNumber'] for i in range(size) ]
Colegios['Código Postal'] = [ firmadas[i]['postalCode'] for i in range(size) ]
Colegios['Ciudad'] = [ firmadas[i]['cityName'] for i in range(size) ]
Colegios['Estado'] = [ firmadas[i]['stateName'] for i in range(size) ]
Colegios['País'] = [ firmadas[i]['countryName'] for i in range(size) ]

Colegios.to_csv('../data/school_data.csv')