# Importacion de librerias necesarias
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests
import json
import matplotlib.pyplot as plt

# Función para cargar datos desde Elasticsearch
def load_data_from_elasticsearch(url, index):
    query = {
        "query": {
            "match_all": {}
        }
    }
    response = requests.get(f'{url}/{index}/_search?size=10000', json=query, auth=('elastic', 'elastic'))
    hits = response.json()['hits']['hits']
    data = [hit['_source'] for hit in hits]
    df = pd.DataFrame(data)
    return df

# Función para crear un índice en Elasticsearch
def create_index(url, index):
    response = requests.put(f'{url}/{index}', auth=('elastic', 'elastic'))
    print(response.text)

# URL de Elasticsearch y nombre del índice
url = 'http://localhost:9200'
index = 'irisdata'

# Carga de datos desde Elasticsearch
df = load_data_from_elasticsearch(url, index)

# Separar características y etiquetas
X = df.drop(columns=['target']).values
y = df['target'].values

# Se crean los índices en Elasticsearch para almacenar los resultados
create_index(url, 'iris_clasificacion')
create_index(url, 'iris_regresion')
create_index(url, 'iris_clustering')
create_index(url, 'iris_pca')

# Clasificación: K vecinos más cercanos (KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Se divide el conjunto de datos en conjuntos de entrenamiento (80%) y prueba (20%).
scaler = StandardScaler()
# Se estandarizan las características para que todas tengan la misma escala
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
# Se crea un clasificador KNN utilizando los 3 vecinos más cercanos para la clasificación.

knn.fit(X_train_scaled, y_train)
# Se entrena el clasificador KNN utilizando el conjunto de datos de entrenamiento escalado (características y etiquetas).

accuracy = knn.score(X_test_scaled, y_test)
# Se evalúa el rendimiento del clasificador utilizando el conjunto de datos de prueba escalado.

print("Accuracy of KNN Classifier:", accuracy)

# Se guardan los resultados de la clasificación en Elasticsearch
url_clasificacion = f'{url}/iris_clasificacion/_doc'
y_test_serializable = y_test.astype(int).tolist()

for i in range(len(X_test)):
    etiqueta_predicha = int(knn.predict(X_test_scaled[i].reshape(1, -1))[0])
    data = {
        "caracteristica_1": X_test[i, 0],
        "caracteristica_2": X_test[i, 1],
        "caracteristica_3": X_test[i, 2],
        "caracteristica_4": X_test[i, 3],
        "etiqueta_real": y_test_serializable[i],
        "etiqueta_predicha": etiqueta_predicha,
        "precision_knn": accuracy
    }
    response = requests.post(url_clasificacion, json=data, auth=('elastic', 'elastic'))
    print(response.text)

# Regresión lineal
print("Columnas del DataFrame para regresión:", df.columns)  # Añadir esta línea para verificar las columnas

X = df[['petal length (cm)']].values   # Tomamos solo la tercera característica (longitud del pétalo)
y = df['petal width (cm)'].values     # La cuarta característica será nuestro objetivo (anchura del pétalo)

# Se divide el conjunto de datos en conjuntos de entrenamiento y prueba igual que antes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de características
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Se crea y entrena el modelo de regresión lineal
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Se hacen predicciones
y_pred = lr.predict(X_test_scaled)

# Se evalua el rendimiento del modelo
r2_score = lr.score(X_test_scaled, y_test)
print("R^2 Score of Linear Regression:", r2_score)

# Los resultados de la regresión son mandados a Elasticsearch
url_regresion = f'{url}/iris_regresion/_doc'

for i in range(len(X_test)):
    data = {
        "longitud_petalo": X_test[i, 0],
        "anchura_petalo_real": y_test[i],
        "anchura_petalo_predicha": y_pred[i],
        "r2_score": r2_score
    }
    response = requests.post(url_regresion, json=data, auth=('elastic', 'elastic'))
    print(response.text)

# Clustering: K-Means
X = df.drop(columns=['target']).values

# Se crea el modelo de clustering K-Means
kmeans = KMeans(n_clusters=3)  # 3 clusters porque hay 3 tipos en el conjunto de datos Iris

# Se entrena el modelo de clustering
kmeans.fit(X)

# Visualización de los clústeres
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.title("Clustering K-Means en el conjunto de datos Iris")
plt.xlabel("Longitud del Sépalo (cm)")
plt.ylabel("Anchura del Sépalo (cm)")
plt.show()

# Los resultados del clustering son mandados a Elasticsearch
url_clustering = f'{url}/iris_clustering/_doc'

for i in range(len(X)):
    data = {
        "longitud_sepalo_cluster": X[i, 0],
        "anchura_sepalo_cluster": X[i, 1],
        "clustering_label": int(kmeans.labels_[i])
    }
    response = requests.post(url_clustering, json=data, auth=('elastic', 'elastic'))
    print(response.text)

# Reducción de dimensionalidad: PCA
X = df.drop(columns=['target']).values
y = df['target'].values

# Se crea una instancia de PCA reduciendo la dimensionalidad a dos componentes principales (gráfico bidimensional)
pca = PCA(n_components=2)

# Se entrena el modelo PCA con las características de entrada
X_pca = pca.fit_transform(X)

# Se muestran los datos reducidos en un gráfico de dispersión 
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title("PCA del Iris Dataset")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# Los datos transformados por PCA son mandados a Elasticsearch
url_pca = f'{url}/iris_pca/_doc'

for i in range(len(X_pca)):
    data = {
        "componente_principal_1": X_pca[i, 0],
        "componente_principal_2": X_pca[i, 1],
        "etiqueta": int(y[i]) 
    }
    response = requests.post(url_pca, json=data, auth=('elastic', 'elastic'))
    print(response.text)
