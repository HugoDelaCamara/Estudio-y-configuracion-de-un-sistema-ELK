# Importacion de librerias necesarias
import pandas as pd
from sklearn import datasets
from elasticsearch import Elasticsearch

# Cargar el dataset de Iris
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Conexión a Elasticsearch 
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Definición del índice
index_name = 'irisdata'

# Campos que va a contener el indice
mapping = {
    "mappings": {
        "properties": {
            "sepal length (cm)": {"type": "float"},
            "sepal width (cm)": {"type": "float"},
            "petal length (cm)": {"type": "float"},
            "petal width (cm)": {"type": "float"},
            "target": {"type": "integer"}
        }
    }
}

# Se crea el índice con el mapeo en caso de que no existiese
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)


# Se cargan los datos en Elasticsearch
for i, row in df.iterrows():
    # Se convierte cada fila en un diccionario
    doc = row.to_dict()
    res = es.index(index=index_name, id=i, body=doc)
    print(res['result'])
