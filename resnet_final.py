import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob
from tensorflow.keras import layers
from tensorflow.keras import layers
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

# center and norm data
images_temp = images - images.mean()
images_norm = images_temp / 255

# ResNet18 architecture

# Residual block. Introduces shortcut connections that allow the gradient to flow more easily during training.
# Such structures represent a solution to the vanishing gradient problem (gradients become extremely small 
# as they are propagated backward through the layers)
def residual_block(x, filters, strides):
    shortcut = x

    # First convolutional layer of the residual block. The output is supposed to have the following shape: 
    # (n, N, N, `filters`), where
    # n = the number of input instances
    # N = 1 + [(`input_volume` - `kernel_size` + 2*`padding`) / `strides`]
    # `padding='same'` adjusts padding such that the output has the same size as the input
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    # re-centre and re-scale data
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # `filters` need to match in order to execute the shortcut connection. In addition, coincidentally, `strides` change to 
    # the value of (2,2), when the filters increase.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the current layer with the shortcut layer to ensure a better gradient flow
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# input shape of the model
input_tensor = tf.keras.Input(shape=(128,128, 1))

x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same')(input_tensor)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

x = residual_block(x, filters=64, strides=(1, 1))
x = residual_block(x, filters=64, strides=(1, 1))
x = residual_block(x, filters=128, strides=(2, 2))
x = residual_block(x, filters=128, strides=(1, 1))
x = residual_block(x, filters=256, strides=(2, 2))
x = residual_block(x, filters=256, strides=(1, 1))
x = residual_block(x, filters=512, strides=(2, 2))
x = residual_block(x, filters=512, strides=(1, 1))

# Collapses the spatial dimensions to a single value per channel
# Since this is the layer we want to extract the features from, we skip creating a dense layer
x = layers.GlobalAveragePooling2D()(x)

# Create and inspect the model
model = tf.keras.Model(inputs=input_tensor, outputs=x)
model.summary()

# Extract features from avg_pool layer.
features = model.predict(images_norm)

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
plt.title('Resnet18 features - HDBSCAN cluster representation')
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
plt.title('Resnet18 features - GMM cluster representation')
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
plt.title('Resnet18 features - K-Means cluster representation')
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