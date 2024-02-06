# Sentence Pair Classification

## Short description

The main task involves the unsupervised classification of MRI images from patients diagnosed with Alzheimer's. The data set can be found at the following address: https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images.

The code was written in Python and the following data preprocessing techniques were used:

- Dimensionality reduction using Principal Component Analysis (PCA).
- Passing data trough a ResNet18 convolutional neural network that was built from scratch using TensorFlow.

The models used in this study are:

- Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN).
- Gaussian Mixture Model (GMM).
- K-Means Clustering.

Finally, the results are compared with a supervised approach, namely Random Forest Classification, and with a dummy classifier that mimics random choice.

In terms of performance metrics, the following were used:

- Silhouette score.
- Fowlkes-Mallows Index (FMI).
- Adjusted Rand Index (ARI).



A visual study of the hyperparameter tuning and the conclusions are presented in `Documentation.pdf`.
