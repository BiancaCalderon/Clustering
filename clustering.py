import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist

# Cargar el dataset
file_path = "movies.csv"
df = pd.read_csv(file_path, delimiter=';', encoding='ISO-8859-1')

# Selección de columnas numéricas relevantes para clustering
columns_to_use = ['budget', 'revenue', 'runtime', 'popularity', 'voteAvg', 'voteCount',
                  'genresAmount', 'productionCoAmount', 'productionCountriesAmount',
                  'actorsAmount', 'castWomenAmount', 'castMenAmount']
data = df[columns_to_use].dropna()

data = data.select_dtypes(include=[np.number])  # Filtrar solo columnas numéricas
# Escalar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Determinar el número óptimo de clusters con la gráfica del codo
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.axvline(x=4, color='r', linestyle='--')  # Línea vertical en el número óptimo de clústeres
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo para determinar k')
plt.savefig("codo_clusters.png")  # Guardar la figura
plt.close()  # Cerrar la figura


# Aplicación de K-Means con el número óptimo de clusters (suponiendo k=4 basado en el codo)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster_kmeans'] = np.nan  # Inicializar la columna con NaN
df.loc[data.index, 'cluster_kmeans'] = kmeans.fit_predict(data_scaled)  # Asignar solo a las filas válidas

# Aplicación de clustering jerárquico
hierarchical = AgglomerativeClustering(n_clusters=4)
df['cluster_hierarchical'] = np.nan  # Inicializar la columna con NaN
df.loc[data.index, 'cluster_hierarchical'] = hierarchical.fit_predict(data_scaled)  # Asignar solo a las filas válidas

# Guardar resultados
df.to_csv("movies_clustered.csv", index=False)
