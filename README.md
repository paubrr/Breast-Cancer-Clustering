# Unsupervised Clustering Analysis - Breast Cancer Wisconsin Dataset
Comparative analysis of three unsupervised clustering algorithms (K-Means, DBSCAN, and BIRCH) applied to the _Breast Cancer Wisconsin (Diagnostic)_ dataset. The goal is to explore whether clinically meaningful patient subgroups (broadly corresponding to benign and malignant cases) can be recovered from cell measurement features alone, without using diagnostic labels.

## Algorithms used: 
| Algorithm | Approach | Key Hyperparameters |
| -------- | -------- | -------- |
| K-Means | Centroid-based |k = 2 (Selected via the Elbow Method |
| DBSCAN | Density-based | eps = 0.2, min_samples = 5 |
| BIRCH | Hierarchical | n_clusters= 2|

## Results 

| Algorithm | Silhouette Score | Agreement with diagnosis |
| -------- | -------- | -------- |
| K-Means | 0.3785 | 91.2% |
| BIRCH | 0.3460 | 90.8% |
| DBSCAN | N/A | N/A |

K-Means achieved the highest silhouette score, followed closely by BIRCH. At the setting tested (eps = 0.2), DBSCAN labelled all 455 training samples as noise and formed no clusters. This highlights a known limitation of density-based methods on high-dimensional biomedical data.


### PCA Cluster Visualization

<img width="2646" height="770" alt="pca_clusters" src="https://github.com/user-attachments/assets/d03e9235-f1d0-4317-a81e-28450521b19e" />


### Elbow Method (K-Means)

<img width="1200" height="750" alt="elbow_method" src="https://github.com/user-attachments/assets/beee21d5-0039-4f9c-889e-29fa9f95eeef" />


### Silhouette Score Comparison

<img width="1050" height="750" alt="silhouette_comparison" src="https://github.com/user-attachments/assets/a0ae6e40-533c-47f1-8b11-be039980812b" />


## Dataset

**Breast Cancer Wisconsin (Diagnostic)** - UCI Machine Learning Repository\
569 instances • 30 numeric features • 2 classes (benign/malignant)\
Labels are not used during training, as this is a completely unsupervised analysis

Loaded directly via sklearn.datasets.load_breast_cancer() -> No download needed

Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 

## How to Run 

```bash
git clone https://github.com/paubrr/Breast-Cancer-Clustering.git
cd Breast-Cancer-Clustering
pip install -r requirements.txt
mkdir figures
python breast_cancer_clustering.py
```

### Dependencies
• numpy
• pandas
• matplotlib
• seaborn
• scikit-learn

## Discussion

The two clusters recovered by K-Means and BIRCH match the benign/malignant diagnosis for about 91% of samples, even though labels were never provided to the models (they were used only afterwards, for evaluation). This suggests that the 30 cell-nucleus measurements carry enough discriminative signal for unsupervised separation. 

DBSCAN's not-so-good result here is very informative, since the data does not form well-separated density regions in the normalized feature space, which is common in clinical tabular datasets where class boundaries are gradual rather than sharp. Future work could explore kernel-based density estimation or manifold learning as preprocessing steps before DBSCAN

------------------------------------------------------------------------------------------------------
 
_Part of the AI with Python Certificate - Universidad Anáhuac Online_\
(2024-2025)\
_Author: Paula Bustos R_
