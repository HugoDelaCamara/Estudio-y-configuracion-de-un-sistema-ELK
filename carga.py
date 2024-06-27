import pandas as pd
from sklearn import datasets
from elasticsearch import Elasticsearch

# Cargar el dataset de Iris
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Conectar a Elasticsearch con el esquema especificado
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Definir el nombre del índice
index_name = 'irisdata'

mapping = {
    "mappings": {
        "properties": {
            "sepal_length": {"type": "float"},
            "sepal_width": {"type": "float"},
            "petal_length": {"type": "float"},
            "petal_width": {"type": "float"},
            "target": {"type": "integer"}
        }
    }
}

# Crear el índice con el mapeo
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mapping)


# Importar los datos en Elasticsearch
for i, row in df.iterrows():
    # Convertir la fila a un diccionario
    doc = row.to_dict()
    
    # Indexar el documento en Elasticsearch
    res = es.index(index=index_name, id=i, body=doc)
    print(res['result'])
