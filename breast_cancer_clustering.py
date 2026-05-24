# Unsupervised Clustering Analysis on Breast Cancer Wisconsin Dataset
# Algorithms: K-Means, DBSCAN, BIRCH
# Paula Bustos R

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('figures', exist_ok=True)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

#Load and Prepare the data

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

#Unsupervised Clustering -> no labels

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Correlation Heatmap

plt.figure(figsize=(14, 10))
sns.heatmap(
    pd.DataFrame(X_train_scaled, columns=df.columns).corr(),
    cmap='coolwarm',
    annot=False,
    linewidths=0.3
)
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=150)
plt.show()

# K-Means (Elbow Method to find best k)

inertia = []
k_values = range(1, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linewidth=2, color='steelblue')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method — K-Means')
plt.xticks(k_values)
plt.tight_layout()
plt.savefig('figures/elbow_method.png', dpi=150)
plt.show()

# Fit final K-Means with k=2 (benign vs malignant)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_scaled)

# DBSCAN

dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_train_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"\nDBSCAN — Clusters found: {n_clusters_dbscan} | Noise points: {n_noise}")

# BIRCH

birch = Birch(n_clusters=2)
birch_labels = birch.fit_predict(X_train_scaled)


# PCA Visualization (All 3 algorithms)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

configs = [
    (kmeans_labels, 'K-Means (k=2)', 'tab10'),
    (dbscan_labels, 'DBSCAN', 'tab10'),
    (birch_labels, 'BIRCH (n=2)', 'viridis'),
]

for ax, (labels, title, cmap) in zip(axes, configs):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.6, s=30)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    plt.colorbar(scatter, ax=ax)

plt.suptitle('Cluster Visualization via PCA (2 Components)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

#Evaluation using Silhouette scores

silhouette_kmeans = silhouette_score(X_train_scaled, kmeans_labels)
silhouette_birch = silhouette_score(X_train_scaled, birch_labels)

if n_clusters_dbscan > 1:
    silhouette_dbscan = silhouette_score(X_train_scaled, dbscan_labels)
else:
    silhouette_dbscan = None
    print("DBSCAN: Not enough clusters to compute silhouette score.")

print("\n" + "="*45)
print("        SILHOUETTE SCORE COMPARISON")
print("="*45)
print(f"  K-Means : {silhouette_kmeans:.4f}")
print(f"  BIRCH   : {silhouette_birch:.4f}")
if silhouette_dbscan is not None:
    print(f"  DBSCAN  : {silhouette_dbscan:.4f}")
else:
    print(f"  DBSCAN  : N/A (insufficient clusters)")
print("="*45)

# Barchart comparison

algorithms = ['K-Means', 'BIRCH']
scores = [silhouette_kmeans, silhouette_birch]

if silhouette_dbscan is not None:
    algorithms.append('DBSCAN')
    scores.append(silhouette_dbscan)

colors = ['steelblue', 'mediumseagreen', 'tomato']
plt.figure(figsize=(7, 5))
bars = plt.bar(algorithms, scores, color=colors[:len(algorithms)], width=0.5, edgecolor='white')
plt.ylim(0, 1)
plt.ylabel('Silhouette Score')
plt.title('Algorithm Comparison — Silhouette Score')
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/silhouette_comparison.png', dpi=150)
plt.show()
