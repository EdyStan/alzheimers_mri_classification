import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, classification_report


# load RGB images and labels
main_folder = '/home/edstan/Desktop/master_AI/pml/final_projects/project_2/Dataset/'
images = []
labels = []
for class_folder in os.listdir(main_folder):
    class_path = os.path.join(main_folder, class_folder)
    local_images_raw = glob(os.path.join(class_path, '*.jpg'))
    local_images = [cv2.imread(img) for img in local_images_raw]
    images.extend(local_images)
    labels.extend([class_folder] * len(local_images_raw))

images_rgb = np.array(images)
labels = np.array(labels)

# plot an image from each category
unique_img = [np.where(labels == label)[0][0] for label in np.unique(labels)]
fig, axis = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in zip(unique_img, [[0,0], [0,1], [1,0], [1,1]]):
    axis[ax[0], ax[1]].imshow(cv2.cvtColor(images_rgb[i], cv2.COLOR_BGR2RGB))
    axis[ax[0], ax[1]].set_title(labels[i])
    axis[ax[0], ax[1]].axis('off')
plt.show()

# since the images are black and white, we decide to remove the RGB dimension and keep them gray
images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images_rgb])

# plot the black and white images
fig, axis = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in zip(unique_img, [[0,0], [0,1], [1,0], [1,1]]):
    axis[ax[0], ax[1]].imshow(cv2.cvtColor(images_gray[i], cv2.COLOR_BGR2RGB))
    axis[ax[0], ax[1]].set_title(labels[i])
    axis[ax[0], ax[1]].axis('off')
plt.show()

images = images_gray

# check the number of instances for each label. It looks like the dataset is imbalanced
unique_img, counts = np.unique(labels, return_counts=True)
for i, j in zip(unique_img, counts):
    print(f"{i}: {j}")


images_flatten = images.reshape(6400, -1)

# Principal Component Analysis (PCA) 
# orders dimensions by the amount of variance they explain
# Black box explained:
# 1. Centers the data by extracting the mean of each feature
# 2. Calculates the covariance matrix (the relationship between each feature)
# 3. Computes eigenvalues and eigenvectors of the covariance matrix
# (eigenvalues can be interpreted as the magnitude of variance, and the eigenvectors
# as the directions of the maximum variance)
# 4. Principal components are chosen based on the amount of variance we want to retain in the dataset
# 5. The original data is projected onto this new space 
# (the centered data matrix is multiplied by the projection matrix)
pca_dimensions = PCA()
pca_dimensions.fit(images_flatten)
# we look for instances with the smallest explained variance (eigenvalue / total variance)
cumulative_variance = np.cumsum(pca_dimensions.explained_variance_ratio_)
# we choose the percentage of variance to be left
variance_left = 0.95
num_features = np.argmax(cumulative_variance >= variance_left) + 1
print(f"Number of features for {variance_left} explained variance: {num_features}")

# As a result of the previous calculations, we compute PCA with the chosen number of principal components
# Moreover, we check if the inverse transformation takes us back to a distinguishable image
pca = PCA(n_components=num_features)
images_reduced = pca.fit_transform(images_flatten)
images_recovered = pca.inverse_transform(images_reduced)

# Show the original image alongside with the decompressed image
images_recovered_show = images_recovered.reshape(6400, 128, 128)
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(images[0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(images_recovered_show[0], cmap='gray')
plt.title('Recovered Image')
plt.axis('off')

plt.show()

# Because PCA already extracted the mean from the data, we only need to norm it
# the reduced data does not range from 0 to 255 anymore. In fact, the values reach 1000+
features = images_reduced / np.max(np.abs(images_reduced))

# Extract the first two components with the highest variance, using PCA, to further represent the clusters
pca = PCA(n_components=2)
features_show = pca.fit_transform(features)


# The metrics we choose to use in order to evaluate the model are:
# -silhouette_score -> indicates how well-separated the clusters are
# it ranges from -1 to 1. A high value indicates that the point is well matched with
# its cluster and poorly matched with the other clusters
# -Fowlkes-Mallows Index (FMI) -> evaluates the similarity between the 
# true labels and the predicted ones. It ranges from 0 to 1 and measures 
# the geometric mean of precision and recall.
# Adjusted Rand Index (ARI) -> measures the similarity between two clustering results
# (true and predicted labels in our case). It specifies the similarity with 
# random chance, ranging from -1 to 1, 1 meaning perfect agreement with the true labels and
# 0 meaning that the predicted labels are similar to random chance. 

# Hierarchical Density-Based Spatial Clustering of Applications with Noise
# identifies clusters based on regions of high data point density
# builds a hierarchy of clusters, organizing them into a tree structure (dendrogram)
# as the algorithm progresses, it forms clusters by merging or splitting existing clusters
# the hierarchy is constructed by considering different density thresholds
# for example, as the density threshold increases, larger clusters are created
hdb = HDBSCAN(min_cluster_size=4)
predicted_labels = hdb.fit_predict(features)
s_score = silhouette_score(features, predicted_labels, metric="euclidean")
fmi_score = fowlkes_mallows_score(labels, predicted_labels)
ari_score = adjusted_rand_score(labels, predicted_labels)
print(s_score, fmi_score, ari_score)

# Plot clusters identified by the model
plt.scatter(features_show[:, 0], features_show[:, 1], c=predicted_labels, cmap='viridis')
plt.title('PCA features - HDBSCAN cluster representation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# Gaussian Mixture Model (GMM)
# used for representing a mixture of multiple Gaussian distributions
# it assumes that the data points are generated from a mixture of several Gaussian distributions
# each associated with a cluster
gmm = GaussianMixture(n_components=4, init_params='random', max_iter=1000, tol=1e-3)
predicted_labels = gmm.fit_predict(features)

s_score = silhouette_score(features, predicted_labels, metric="euclidean")
fmi_score = fowlkes_mallows_score(labels, predicted_labels)
ari_score = adjusted_rand_score(labels, predicted_labels)
print(s_score, fmi_score, ari_score)

# Plot clusters identified by the model
plt.scatter(features_show[:, 0], features_show[:, 1], c=predicted_labels, cmap='viridis')
plt.title('PCA features - GMM cluster representation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# K-Means clustering
clust = KMeans(n_clusters=4, n_init=10)
predicted_labels = clust.fit_predict(features)

s_score = silhouette_score(features, predicted_labels, metric="euclidean")
fmi_score = fowlkes_mallows_score(labels, predicted_labels)
ari_score = adjusted_rand_score(labels, predicted_labels)
print(s_score, fmi_score, ari_score)

# Plot clusters identified by the model
plt.scatter(features_show[:, 0], features_show[:, 1], c=predicted_labels, cmap='viridis')
plt.title('PCA features - K-Means cluster representation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# the supervised learning algorithm to compare with
X_train, X_test, y_train, y_test = train_test_split(features, labels)
clf = RandomForestClassifier()
clf.fit(X_train, y_train) 
predicted_labels = clf.predict(X_test)
classification_rep = classification_report(y_test, predicted_labels)
print(classification_rep)


# DummyClassifier for comparison with random
clf = DummyClassifier(strategy='uniform')
clf.fit(X_train, y_train) 
predicted_labels = clf.predict(X_test)
classification_rep = classification_report(y_test, predicted_labels)
print(classification_rep)