Practical 1
Data Augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt

# Load sample image
img = load_img("Cat.png")
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

# Basic augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# Generate 5 augmented samples
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.imshow(batch[0].astype("uint8"))
    plt.axis("off")
    plt.show()
    
    i += 1
    if i == 5:
        break

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import random

# Load image
img = Image.open("Cat.png").convert("RGB")

# Function for simple augmentations
def augment_image(img):
    aug = img.copy()

    # Random rotation (-20° to +20°)
    angle = random.uniform(-20, 20)
    aug = aug.rotate(angle)

    # Random horizontal flip
    if random.random() > 0.5:
        aug = ImageOps.mirror(aug)

    # Random zoom (crop + resize)
    zoom_factor = random.uniform(0.8, 1.0)
    w, h = aug.size
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    aug = aug.crop((left, top, left + new_w, top + new_h))
    aug = aug.resize((w, h))

    return aug

# Generate augmented images
plt.figure(figsize=(8,8))

for i in range(6):
    aug_img = augment_image(img)
    plt.subplot(2,3,i+1)
    plt.imshow(aug_img)
    plt.axis("off")

plt.show()

*ADVANCE*
import tensorflow as tf
import matplotlib.pyplot as plt

def advanced_augment(image):
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, 0.8, 1.5)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[180, 180, 3])  # crop
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image

img = tf.image.decode_jpeg(tf.io.read_file("Cat.png"))
img = tf.image.resize(img, [224, 224]) / 255.0

aug = advanced_augment(img)
plt.imshow(aug)
plt.axis("off")
plt.show()

PRACTICAL2 TRANSFER LEARNING
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def filter_class(X, y):
    idx = np.where((y == 3) | (y == 5))[0]
    X = X[idx]
    y = (y[idx] == 5).astype(int).ravel()   # dog=1, cat=0
    return X, y

X_cd_train, y_cd_train = filter_class(x_train, y_train)
X_cd_test, y_cd_test = filter_class(x_test, y_test)

def preprocess_batch(X):
    out = []
    for img in X:
        gray = rgb2gray(img)
        small = resize(gray, (32, 32))
        out.append(small.flatten())
    return np.array(out)

X_train_p = preprocess_batch(X_cd_train)
X_test_p = preprocess_batch(X_cd_test)

# clf = LogisticRegression(max_iter=2000)
# clf.fit(X_train_p, y_cd_train)
# achieved accuracy of Accuracy: 0.5665

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_p, y_cd_train)

# achieved accuracy of Accuracy: 0.656

# -------- 5) Evaluate --------
pred = clf.predict(X_test_p)
print("Accuracy:", accuracy_score(y_cd_test, pred))

Accuracy: 0.656


# -------- 6) Predict on your own image --------
img_path = "Cat.png"

img = Image.open(img_path).convert("RGB")
img = np.array(img)

gray = rgb2gray(img)
small = resize(gray, (32, 32)).flatten()

prediction = clf.predict([small])[0]
print("Prediction:", "DOG" if prediction == 1 else "CAT")

Prediction: CAT

PRACTICAL 3 FEW SHOT LEARNING
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

data = load_breast_cancer()
X = data.data
y = data.target
classes = data.target_names   # ['malignant', 'benign']

print("\nTotal Columns:", len(data.feature_names))
print("Column Names:")
for col in data.feature_names:
    print("-", col)

df = pd.DataFrame(X, columns=data.feature_names)
print("\nSample Data:")
print(df)

X_few = []
y_few = []

for c in np.unique(y):
    idx = np.where(y == c)[0][:3]   # 3-shot
    X_few.append(X[idx])
    y_few.append(y[idx])

X_few = np.vstack(X_few)
y_few = np.hstack(y_few)

print("\nFew-shot samples used:", len(X_few))

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_few, y_few)

Out[5]:
KNeighborsClassifier
KNeighborsClassifier(n_neighbors=1)

pred_all = model.predict(X)
print("\nAccuracy on FULL dataset:", accuracy_score(y, pred_all))

sample = X[451].reshape(1, -1)
pred = model.predict(sample)[0]
print("\nPrediction on real dataset sample (X[451]):", classes[pred])

new_data = np.array([
    14.5, 20.1, 95.0, 600.0, 0.11, 0.12, 0.09, 0.06, 0.20, 0.07,
    0.20, 1.10, 1.30, 8.0, 0.006, 0.01, 0.02, 0.007, 0.015, 0.003,
    16.0, 25.0, 110.0, 800.0, 0.14, 0.18, 0.15, 0.08, 0.27, 0.09
])

pred_all = model.predict(X)
print("\nAccuracy on FULL dataset:", accuracy_score(y, pred_all))


practical 4 reductioning using pca
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()
X = data.data          # (569, 30)
y = data.target
feature_names = data.feature_names

print("Original shape:", X.shape)

df = pd.DataFrame(X, columns=feature_names)

plt.figure(figsize=(18, 15))
df.hist(bins=20, color="steelblue", edgecolor="black", figsize=(18, 15))
plt.suptitle("Histograms of All Features (Breast Cancer)", fontsize=16)
plt.tight_layout()
plt.show()

# Standardize (important for PCA)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# PCA -> 2 components

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("Reduced shape:", X_pca.shape)


# Explained variance

print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)
print("Total variance captured by 2 PCs:", pca.explained_variance_ratio_.sum())


plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', s=40, alpha=0.7)
plt.xlabel("PC1 (most variance)")
plt.ylabel("PC2 (2nd most variance)")
plt.title("PCA Projection (2D) - Breast Cancer Dataset")
plt.legend(*scatter.legend_elements(), title="Classes")
plt.grid(alpha=0.3)
plt.show()


practical 5 singular value decomposition
from sklearn.datasets import load_wine
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# 1) Load dataset
data = load_wine()
X = data.data
y = data.target
print("Original shape:", X.shape)   # (178, 13)

# 2) Apply SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X)

print("Reduced shape:", X_svd.shape)
print("Explained variance:", svd.explained_variance_ratio_)

# 3) Visualize SVD result
plt.scatter(X_svd[:,0], X_svd[:,1], c=y, cmap='viridis', s=45)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVD Projection (Wine Dataset)")
plt.colorbar(label="Class label")
plt.show()


ptrtactical 6 partition based clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Load dataset
data = load_iris()
X = data.data        # shape (150,4)
names = data.feature_names
names

# Standardize (important for distance-based clustering)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Elbow method to pick k (compute inertia for k=1..8)
inertias = []
K_range = range(1, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(Xs)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(6,3))
plt.plot(K_range, inertias, '-o')
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (sum of squared distances)")
plt.title("Elbow method")
plt.xticks(K_range)
plt.grid(alpha=0.3)
plt.show()

# Silhouette score for k=2..6
sil_scores = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)
    sil_scores.append(silhouette_score(Xs, labels))

plt.figure(figsize=(6,3))
plt.plot(range(2,7), sil_scores, '-o')
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs k")
plt.grid(alpha=0.3)
plt.show()


# Fit final KMeans (choose k=3 from domain knowledge / elbow)
k_final = 3
kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=20)
labels = kmeans.fit_predict(Xs)


# Fast variant: MiniBatchKMeans
mbk = MiniBatchKMeans(n_clusters=k_final, random_state=42, batch_size=20)
mbk_labels = mbk.fit_predict(Xs)


# Visualize clusters in 2D using PCA projection
pca = PCA(n_components=2, random_state=42)
Xp = pca.fit_transform(Xs)

plt.figure(figsize=(7,5))
scatter = plt.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=60, alpha=0.8)

centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c='black', s=120, marker='x', label='centroids')

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(f"KMeans clusters (k={k_final}) projected to 2D")
plt.legend()
plt.grid(alpha=0.2)

plt.show()

# Evaluate: silhouette score for final clustering
print("KMeans inertia:", kmeans.inertia_)
print("Silhouette score (kmeans):", silhouette_score(Xs, labels))
print("Silhouette score (minibatch):", silhouette_score(Xs, mbk_labels))

# Predict cluster for a new sample (give raw feature values -> standardized)
new_sample = np.array([[5.0, 3.2, 1.2, 0.2]])  # example in original feature values
new_sample_std = scaler.transform(new_sample)

pred_cluster = kmeans.predict(new_sample_std)[0]

print("New sample assigned to cluster:", pred_cluster)


practical 7 density based clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.8, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

print("Cluster labels:", np.unique(labels))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="tab10", s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clustering (Density-Based)")
plt.show()


practical 8 hierarchical clustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data points")
plt.ylabel("Distance")
plt.show()

hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X_scaled)

print("Cluster labels:", np.unique(labels))

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', s=40)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clustering (Agglomerative)")
plt.show()


practical 9 market basket analysimng using arpiori algorithm
import pandas as pd

# Sample market basket transactions
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Beer', 'Bread'],
    ['Milk', 'Bread', 'Butter', 'Beer'],
    ['Bread', 'Butter'],
    ['Milk', 'Beer'],
    ['Milk', 'Bread'],
    ['Butter', 'Beer'],
    ['Milk', 'Bread', 'Butter']
]

df = pd.DataFrame(dataset, columns=['Item1', 'Item2', 'Item3', 'Item4'])
print(df)

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(dataset).transform(dataset)

df_encoded = pd.DataFrame(te_array, columns=te.columns_)
print(df_encoded)

from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    df_encoded,
    min_support=0.3,
    use_colnames=True
)

print(frequent_itemsets)

from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])


practical 10 fp growth algorithm
import pandas as pd

# Each list represents symptoms observed in one patient
transactions = [
    ['Fever', 'Cough', 'Headache'],
    ['Fever', 'Cough'],
    ['Cough', 'Shortness of Breath'],
    ['Fever', 'Headache'],
    ['Cough', 'Chest Pain'],
    ['Fever', 'Cough', 'Chest Pain'],
    ['Headache', 'Nausea'],
    ['Fever', 'Cough', 'Shortness of Breath'],
    ['Cough', 'Headache'],
    ['Fever', 'Nausea'],
    ['Chest Pain', 'Shortness of Breath'],
    ['Fever', 'Cough', 'Headache'],
    ['Cough', 'Nausea'],
    ['Fever', 'Chest Pain'],
    ['Cough', 'Shortness of Breath', 'Chest Pain']
]

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)
print(df.head())

from mlxtend.frequent_patterns import fpgrowth

frequent_itemsets = fpgrowth(
    df,
    min_support=0.3,
    use_colnames=True
)

print(frequent_itemsets.sort_values('support', ascending=False))

from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.6
)

rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
print(rules.sort_values('lift', ascending=False))
