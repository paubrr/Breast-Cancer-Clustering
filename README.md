# Unsupervised Clustering Analysis - Breast Cancer Wisconsin Dataset
Comparative analysis of three unsupervised clustering algorithms (K-Means, DBSCAN, and BIRCH) applied to the _Breast Cancer Wisconsin (Diagnostic)_ dataset. The goal is to explore whether clinically meaningful patient subgroups (broadly corresponding to benign and malignant cases) can be recovered from cell measurement features alone, without using diagnostic labels.

## Algorithms used: 
| Algorithm | Approach | Key Hyperparameters |
| -------- | -------- | -------- |
| K-Means | Centroid-based |k = 2 (Selected via the Elbow Method |
| DBSCAN | Density-based | eps = 0.2, min_samples = 5 |
| BIRCH | Hierarchical | n_clusters= 2|

## Results 

| Algorithm | Silhouette Score |
| -------- | -------- |
| K-Means | 0.3785 |
| BIRCH | 0.3460 |
| DBSCAN | N/A |

K-Means achieved the highest silhouette score, followed closely by BIRCH. DBSCAN was unable to form any meaningful clusters on this dataset. This is worth noting because the feature space of the Wisconsin dataset is not really density-separable at the scales tested, regardless of the epsilon tuning. This highlights a known limitation of density-based methods on high-dimensional biomedical data.


### PCA Cluster Visualization

### Elbow Method (K-Means)

### Silhouette Score Comparison


## Dataset

**Breast Cancer Wisconsin (Diagnostic)** - UCI Machine Learning Repository\
569 instances • 30 numeric features • 2 classes (benign/malignant)\
Labels are not used during training, as this is a completely unsupervised analysis

Loaded directly via sklearn.datasets.load_breast_cancer() -> No download needed, 

Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 

## How to Run 

**1. Clone the repository**\
git clone https://github.com/paubrr/breast-cancer-clustering.git

**2. Install dependencies**\
pip install -r requirements.txt

**3. Create figures folder**\
mkdir figures

**4. Run**\
python breast_cancer_clustering.py

### Dependencies
• numpy
• pandas
• matplotlib
• seaborn
• scikit-learn

## Discussion

  The two-cluster structure recovered by K-Means and BIRCH loosely mirrors the bening/malignant split present in the ground-truth labels, even though labels were never provided to the models. This suggests that the 30-cell nucleus measurements carry enough discriminative signal for unsupervised separation. 

DBSCAN's not-so-good result here is very informative, since the data does not form well-separated density regions in the normalized feature space, which is common in clinical tabular datasets where class boundaries are gradual rather than sharp. Future work could explore kernel-based density estimation or manifold learning as preprocessing steps before DBSCAN

------------------------------------------------------------------------------------------------------
 
_Part of the AI with Python Certificate (2024-2025) - Universidad Anáhuac Online_\
_Author: Paula Bustos R_
