import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('iris.csv')

# Print dataset information to understand its structure
print("Data columns:", data.columns.tolist())
print("First few rows:")
print(data.head())

# Separate features from target (species)
# Use only numeric columns for clustering
if 'species' in data.columns:
    # If there's a column named 'species'
    feature_columns = data.drop('species', axis=1)
    species = data['species']
else:
    # Otherwise, assume the last column contains species
    feature_columns = data.iloc[:, :-1]  # All columns except the last one
    species = data.iloc[:, -1]  # Last column

# Normalize the numeric features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(feature_columns)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)

# Get clustering results
clusters = kmeans.labels_
centers = kmeans.cluster_centers_

# Visualization using first two features
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.8)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=100)
plt.title('K-means Clustering of Iris Dataset')
plt.xlabel('Feature 1 (normalized)')
plt.ylabel('Feature 2 (normalized)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Compare the clustering results with actual species
if len(species.unique()) <= 3:  # If we have up to 3 species (as expected in Iris)
    result_df = pd.DataFrame({
        'Cluster': clusters,
        'Species': species
    })
    
    # Count species distribution in each cluster
    print("\nSpecies distribution in each cluster:")
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