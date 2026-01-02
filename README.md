## Overview

This project was developed as part of a university assignment for a Computational Linear Algebra / Data Analysis course.  
It applies **Principal Component Analysis (PCA)** and **k-Means clustering** to a real-world survey dataset in order to explore latent behavioral patterns and group similarities.

The workflow covers the complete unsupervised learning pipeline, from preprocessing to interpretation and evaluation.

## Main Steps

- Data preprocessing and encoding of categorical variables
- Feature scaling (MinMaxScaler / StandardScaler comparison)
- Principal Component Analysis (PCA)
  - Explained variance analysis
  - Interpretation and naming of principal components
- k-Means clustering on PCA-reduced data
  - Optimal number of clusters selection using silhouette score
  - Visualization of score plots and centroids
  - Semantic interpretation and naming of clusters
- Cluster evaluation
  - **Internal evaluation** using silhouette analysis
  - **External evaluation** using demographic and behavioral labels (e.g. age, education, smoking, internet usage)

## Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Structure

- `HWpca_Khatibi.ipynb` — main notebook containing all analyses and visualizations
- `responses_hw.csv` — survey responses dataset
- `columns_hw.csv` — metadata and column descriptions

## Notes

This project focuses on **interpretability and methodological correctness** rather than predictive modeling.  
All analyses are reproducible and follow the assignment’s predefined structure.

