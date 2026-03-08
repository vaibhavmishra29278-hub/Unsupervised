PRACTICAL 1 Data Augumation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

# Load image
img = load_img("Cat.png")
img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)

# Define augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# Generate images
count = 0
for batch in datagen.flow(img_array, batch_size=1):

    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    plt.show()

    count += 1
    if count == 5:
        break

        from PIL import Image, ImageOps
import random

img = Image.open("Cat.png").convert("RGB")

def augment_image(img):

    aug = img.copy()

    # Random rotation
    angle = random.uniform(-20, 20)
    aug = aug.rotate(angle)

    # Random flip
    if random.random() > 0.5:
        aug = ImageOps.mirror(aug)

    # Random zoom
    zoom = random.uniform(0.8, 1.0)
    w, h = aug.size
    new_w, new_h = int(w * zoom), int(h * zoom)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    aug = aug.crop((left, top, left + new_w, top + new_h))
    aug = aug.resize((w, h))

    return aug

# Generate samples
for i in range(5):

    augmented = augment_image(img)

    plt.imshow(augmented)
    plt.axis("off")
    plt.show()

    import tensorflow as tf

def advanced_augment(image):

    image = tf.image.random_brightness(image, 0.3)
    image = tf.image.random_contrast(image, 0.8, 1.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[180, 180, 3])

    noise = tf.random.normal(shape=tf.shape(image), stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return image


img = tf.image.decode_image(tf.io.read_file("Cat.png"))
img = tf.image.resize(img, [224, 224]) / 255.0

augmented = advanced_augment(img)

plt.imshow(augmented.numpy())
plt.axis("off")
plt.show()

PRACTICAL 2 TRANSFER LEARNING
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

(x_train, y_train), (x_test,y_test)= keras.datasets.cifar10.load_data()

cat_dog_idx_train = np.where((y_train ==3) | (y_train ==5))[0]
cat_dog_idx_test = np.where((y_test ==3) | (y_test ==5))[0]

x_train = x_train[cat_dog_idx_train]
y_train = (y_train[cat_dog_idx_train]== 5).astype(int) # dog=1, cat= 0

x_test = x_test[cat_dog_idx_test]
y_test = (y_test[cat_dog_idx_test]==5).astype(int) # dog=1, cat= 0

# Resize to 224x224 for MobileNetV2
x_train = tf.image.resize(x_train,(224,224)) / 255.0
x_test = tf.image.resize(x_test,(224, 224)) / 255.0

base = keras.applications.MobileNetV2(
    include_top=False, input_shape=(224,224,3), weights="imagenet")
base.trainable = False

model = keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")   # Binary: cat & dog
])

model.compile(optimizer="adam",
             loss="binary_crossentropy",
             metrics=["accuracy"])

from PIL import Image
import numpy as np

#-------Load NEW image-
img_path = r"C:\Users\User35\Downloads\download (1).jfif"

img = Image.open(img_path).convert("RGB")
img = img.resize((224,224))

x = np.array(img)/ 255.0
x = np.expand_dims(x,0)     # shape(1, 224, 224,3)

# ------- Predict------
prob = model.predict(x)[0][0]

if prob > 0.5:
    print("Prediction: DOG Confidence:", prob)
else:
    print("Prediction: CAT Confidence:", 1-prob)

PRACTICAL 3 FEW SHOT LEARNING
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
classes = data.target_names

# Show dataset info
print("Total Features:", len(data.feature_names))
print("Feature Names:", data.feature_names)

df = pd.DataFrame(X, columns=data.feature_names)
print("\nSample Data:")
print(df.head())

# Few-shot sampling (3 samples per class)
X_few = []
y_few = []

for c in np.unique(y):

    idx = np.where(y == c)[0][:3]

    X_few.append(X[idx])
    y_few.append(y[idx])

X_few = np.vstack(X_few)
y_few = np.hstack(y_few)

print("\nFew-shot samples used:", len(X_few))

# Train model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_few, y_few)

# Evaluate on full dataset
pred_all = model.predict(X)

print("\nAccuracy on full dataset:",
      accuracy_score(y, pred_all))

# Predict existing sample
sample = X[451].reshape(1, -1)

pred = model.predict(sample)[0]

print("\nPrediction for dataset sample:", classes[pred])

# Predict manual input
new_data = np.array([
14.5,20.1,95.0,600.0,0.11,0.12,0.09,0.06,0.20,0.07,
0.20,1.10,1.30,8.0,0.006,0.01,0.02,0.007,0.015,0.003,
16.0,25.0,110.0,800.0,0.14,0.18,0.15,0.08,0.27,0.09
]).reshape(1,-1)

pred2 = model.predict(new_data)[0]

print("\nPrediction for manual input:", classes[pred2])

PRACTICAL 4 DIMENSIONALITY REDUCTIONING USE PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
print("Original shape:", X.shape)

# DataFrame + feature distribution
df = pd.DataFrame(X, columns=data.feature_names)
df.hist(bins=20, figsize=(15,12))
plt.suptitle("Feature Distribution")
plt.show()

# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced shape:", X_pca.shape)
print("Variance captured:", pca.explained_variance_ratio_.sum())

# PCA visualization
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.grid(True)
plt.show()

PRACTICAL 5 SINGULAR VALUE DECOMPOSITION
from sklearn.datasets import load_wine
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X, y = data.data, data.target
print("Original shape:", X.shape)

# Apply SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

print("Reduced shape:", X_svd.shape)
print("Variance captured:", svd.explained_variance_ratio_.sum())

# Visualization
plt.scatter(X_svd[:,0], X_svd[:,1], c=y, cmap="viridis")
plt.xlabel("SVD1")
plt.ylabel("SVD2")
plt.title("SVD Projection (Wine Dataset)")
plt.grid(True)
plt.show()

PRACTICAL 6 PARTITION BASED CLUSTERING
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Load data
X = load_iris().data
Xs = StandardScaler().fit_transform(X)
print("Shape:", X.shape)

# Elbow method
inertia = []
for k in range(1,9):
    inertia.append(KMeans(n_clusters=k, random_state=42).fit(Xs).inertia_)

plt.plot(range(1,9), inertia, '-o')
plt.title("Elbow Method")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.show()

# Silhouette scores
scores = []
for k in range(2,7):
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(Xs)
    scores.append(silhouette_score(Xs, labels))

plt.plot(range(2,7), scores, '-o')
plt.title("Silhouette vs k")
plt.xlabel("k"); plt.ylabel("Score")
plt.show()

# Final clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(Xs)

# MiniBatch KMeans
mbk_labels = MiniBatchKMeans(n_clusters=3, random_state=42).fit_predict(Xs)

# PCA visualization
Xp = PCA(n_components=2).fit_transform(Xs)

plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap="tab10")
centers = PCA(n_components=2).fit(Xs).transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], marker="x")
plt.title("KMeans Clusters")
plt.show()

# Evaluation
print("Silhouette (KMeans):", silhouette_score(Xs, labels))
print("Silhouette (MiniBatch):", silhouette_score(Xs, mbk_labels))

# Predict new sample
sample = StandardScaler().fit(X).transform([[5.0,3.2,1.2,0.2]])
print("Cluster:", kmeans.predict(sample)[0])

PRACTICAL 7 DENSITY BASED CLUSTERING
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load data
X = load_iris().data
Xs = StandardScaler().fit_transform(X)

# DBSCAN clustering
labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(Xs)
print("Clusters:", np.unique(labels))

# PCA for visualization
Xp = PCA(n_components=2).fit_transform(Xs)

plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clusters")
plt.show()

PRACTICAL 8 HIERICHERAL CLUSTERING
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

# Load data
X = load_iris().data
Xs = StandardScaler().fit_transform(X)

# Dendrogram
linked = linkage(Xs, method="ward")
dendrogram(linked, truncate_mode="level", p=5)
plt.title("Dendrogram")
plt.show()

# Agglomerative clustering
labels = AgglomerativeClustering(n_clusters=3).fit_predict(Xs)
print("Clusters:", np.unique(labels))

# PCA visualization
Xp = PCA(n_components=2).fit_transform(Xs)

plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clusters")
plt.show()

PTRACTICAL 9 ARPIORI ALOGORITHM MARKET BASKET
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")

# Transactions
data = [
    ['Milk','Bread','Butter'],
    ['Beer','Bread'],
    ['Milk','Bread','Butter','Beer'],
    ['Bread','Butter'],
    ['Milk','Beer'],
    ['Milk','Bread'],
    ['Butter','Beer'],
    ['Milk','Bread','Butter']
]

# One-hot encoding
te = TransactionEncoder()
df = pd.DataFrame(te.fit(data).transform(data), columns=te.columns_)
print(df)

# Frequent itemsets
freq = apriori(df, min_support=0.3, use_colnames=True)
print(freq)

# Association rules
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])

PRACTICAL 10 FP GROWTH ALGORITHM
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import warnings
warnings.filterwarnings("ignore")
# Symptom transactions
data = [
['Fever','Cough','Headache'],
['Fever','Cough'],
['Cough','Shortness of Breath'],
['Fever','Headache'],
['Cough','Chest Pain'],
['Fever','Cough','Chest Pain'],
['Headache','Nausea'],
['Fever','Cough','Shortness of Breath'],
['Cough','Headache'],
['Fever','Nausea'],
['Chest Pain','Shortness of Breath'],
['Fever','Cough','Headache'],
['Cough','Nausea'],
['Fever','Chest Pain'],
['Cough','Shortness of Breath','Chest Pain']
]

# Encode transactions
te = TransactionEncoder()
df = pd.DataFrame(te.fit(data).transform(data), columns=te.columns_)
print(df.head())

# Frequent itemsets
freq = fpgrowth(df, min_support=0.3, use_colnames=True)
print(freq.sort_values("support", ascending=False))

# Association rules
rules = association_rules(freq, metric="confidence", min_threshold=0.6)
print(rules[['antecedents','consequents','support','confidence','lift']])
