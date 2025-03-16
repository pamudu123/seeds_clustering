# Seeds Dataset Clustering Analysis

## Introduction

The [Seeds Dataset](https://archive.ics.uci.edu/dataset/236/seeds) contains measurements of geometrical properties of wheat kernels belonging to three different varieties: Kama, Rosa, and Canadian. Each instance in the dataset consists of 7 features measured from wheat kernel images:

1. Area
2. Perimeter
3. Compactness
4. Length of kernel
5. Width of kernel
6. Asymmetry coefficient
7. Length of kernel groove

The dataset comprises 210 samples, with 70 samples for each wheat variety. 

## Clustering Methods

### Elbow Method for Determining Optimal Number of Clusters

### Centroid-based Clustering: K-Means

K-Means is a partitioning clustering algorithm that divides data into K distinct, **non-overlapping** clusters.

![K-Means](resources\elbow_method.png)

### Agglomerative Hierarchical Clustering

Hierarchical clustering builds a tree of clusters by progressively merging or splitting groups. Agglomerative clustering follows a bottom-up approach:


#### Different Linkage Methods Analysis

1. **Single Linkage**
   - Defines distance between clusters as the minimum distance between any two points from each cluster

2. **Complete Linkage**
   - Defines distance between clusters as the maximum distance between any two points from each cluster

3. **Average Linkage**
   - Defines distance between clusters as the average distance between all pairs of points across clusters

4. **Ward Linkage**
   - Minimizes the increase in the sum of squared differences within all clusters after merging

### Dendrogram Analysis

![Dendrogram](resources\dendrogram.png)

A dendrogram is a tree-like diagram that records the sequences of merges in hierarchical clustering. It visualizes:

- The hierarchical relationship between clusters
- The distance or dissimilarity between merged clusters (height of the branch)
- The order in which clusters are formed

By cutting the dendrogram horizontally at a certain height, we can obtain any number of clusters.

## Evaluation Metrics

### Internal Evaluation: Silhouette Score

The silhouette score measures how similar an object is to its own cluster compared to other clusters.

### External Evaluation: Purity Score

Purity measures how homogeneous each cluster is with respect to the true classes. 

### Classification Metrics

Since the Seeds dataset includes true labels, we can evaluate clustering as if it were a classification task.

`Clustering + Labels --> Classification`

- Accuracy
- Precision
- Recall
- F1 Score 
- Confusion Matrix
<!-- 
![Dendrogram](resources\dendrogram.png)
![ConfusionMatrix](resources\confusion_matrix.png)
![Visualization_1](resources\visualization_1.png)
![VisualizationBarplot](resources\visualization_barplot.png) -->

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .column {
            flex: 1 1 45%;
            max-width: 45%;
            margin: 5px;
            box-sizing: border-box;
            text-align: center;
        }
        .equal-size {
            width: 300px; /* Set your desired width */
            height: 200px; /* Set your desired height */
            object-fit: contain; /* Maintains aspect ratio */
            display: block;
            margin: 0 auto;
        }
    </style>
    <title>Two-Column Image Layout</title>
</head>
<body>
    <div class="container">
        <div class="column">
            <img src="resources/dendrogram.png" alt="Dendrogram" class="equal-size">
            <img src="resources/visualization_1.png" alt="Visualization 1" class="equal-size">
        </div>
        <div class="column">
            <img src="resources/confusion_matrix.png" alt="Confusion Matrix" class="equal-size">
            <img src="resources/visualization_barplot.png" alt="Visualization Barplot" class="equal-size">
        </div>
    </div>
</body>
</html>