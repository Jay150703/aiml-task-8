# task8_kmeans.py
"""
Task 8: Clustering with K-Means
Objective: Perform unsupervised learning with K-Means clustering.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------
# 1. Load Dataset
# --------------------------
# Example dataset: Mall Customer Segmentation Dataset
# (You can replace with your dataset CSV file)
try:
    df = pd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    print("Dataset not found. Please place 'Mall_Customers.csv' in the same folder.")
    exit()

print("Dataset shape:", df.shape)
print(df.head())

# Use only relevant features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# --------------------------
# 2. Elbow Method
# --------------------------
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# --------------------------
# 3. Apply K-Means
# --------------------------
optimal_k = 5  # From elbow method (adjust if different)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

df['Cluster'] = y_kmeans

# --------------------------
# 4. Visualize Clusters
# --------------------------
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'],
            c=y_kmeans, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means Clustering)")
plt.legend()
plt.show()

# --------------------------
# 5. Silhouette Score
# --------------------------
score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score: {score:.3f}")

# --------------------------
# 6. PCA Visualization (if more than 2 features are used)
# --------------------------
if X.shape[1] > 2:
    pca = PCA(n_components=2)
    reduced_X = pca.fit_transform(X)
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y_kmeans, cmap='viridis', s=50)
    plt.title("K-Means Clustering (PCA Reduced to 2D)")
    plt.show()
