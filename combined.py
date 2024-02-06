from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import numpy as np
import os
from glob import glob
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import fowlkes_mallows_score, silhouette_score, adjusted_rand_score, classification_report
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split


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
resnet18_result = model.predict(images_norm)

images_flatten = resnet18_result.reshape(6400, -1)
pca_dimensions = PCA()
pca_dimensions.fit(images_flatten)
# we look for instances with the smallest explained variance (eigenvalue / total variance)
cumulative_variance = np.cumsum(pca_dimensions.explained_variance_ratio_)
# we choose the percentage of variance to be left
variance_left = 0.98
num_features = np.argmax(cumulative_variance >= variance_left) + 1
print(f"Number of features for {variance_left} explained variance: {num_features}")

# As a result of the previous calculations, we compute PCA with the chosen number of principal components
# Moreover, we check if the inverse transformation takes us back to a distinguishable image
pca = PCA(n_components=num_features)
images_reduced = pca.fit_transform(images_flatten)
features = images_reduced / np.max(np.abs(images_reduced))

X_train, X_test, y_train, y_test = train_test_split(features, labels)
clf = RandomForestClassifier()
clf.fit(X_train, y_train) 
predicted_labels = clf.predict(X_test)
classification_rep = classification_report(y_test, predicted_labels)
print(classification_rep)
