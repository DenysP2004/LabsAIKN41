import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv('iris.csv')

# Вивід інформації про набір даних для розуміння його структури
print("Data columns:", data.columns.tolist())
print("First few rows:")
print(data.head())

# Відокремлення ознак від цільової змінної (виду)
# Використання лише числових стовпців для кластеризації
if 'species' in data.columns:
    # Якщо є стовпець з назвою 'species'
    feature_columns = data.drop('species', axis=1)
    species = data['species']
else:
    # Інакше припускаємо, що останній стовпець містить види
    feature_columns = data.iloc[:, :-1]  # Усі стовпці крім останнього
    species = data.iloc[:, -1]  # Останній стовпець

# Нормалізація числових ознак
scaler = StandardScaler()
data_scaled = scaler.fit_transform(feature_columns)

# Застосування кластеризації K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# Отримання результатів кластеризації
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

# Візуалізація з використанням перших двох ознак
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.8)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100)
plt.title('K-means Кластеризація набору даних Iris')
plt.xlabel('Ознака 1 (нормалізована)')
plt.ylabel('Ознака 2 (нормалізована)')
plt.colorbar(scatter, label='Кластер')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Порівняння результатів кластеризації з реальними видами
if len(species.unique()) <= 3:  # Якщо маємо до 3 видів (як очікується в Iris)
    result_df = pd.DataFrame({
        'Cluster': clusters,
        'Species': species
    })
    
    # Підрахунок розподілу видів у кожному кластері
    print("\nРозподіл видів у кожному кластері:")
    print(pd.crosstab(result_df['Cluster'], result_df['Species']))

from scipy.cluster.hierarchy import dendrogram, linkage
# Ієрархічна кластеризація
Z = linkage(data_scaled, 'ward')
# Дендрограма

dendrogram(Z)
plt.show()

from sklearn.cluster import DBSCAN
# Алгоритм DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data_scaled)
# Результати кластеризації
clusters = dbscan.labels_
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters)
plt.show()