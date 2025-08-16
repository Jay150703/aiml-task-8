Task 8: Clustering with K-Means
Objective

Perform unsupervised learning with K-Means clustering and evaluate results.

Tools Used

Scikit-learn

Pandas

Matplotlib

Steps

Load Dataset

Example: Mall Customer Segmentation Dataset.

Preprocess the dataset (drop unnecessary columns, handle missing values).

Fit K-Means

Apply KMeans from sklearn.cluster.

Assign cluster labels to each data point.

Elbow Method

Run KMeans with different K values.

Plot inertia (within-cluster sum of squares) vs K.

Choose the K where the curve “elbows.”

Cluster Visualization

Reduce to 2D (using PCA if needed).

Scatter plot with cluster colors.

Evaluate Clustering

Compute Silhouette Score to measure clustering quality.
