import keras
import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Clustering: Grouping similar things together WITHOUT being told what they are!
# K-Means: Finds K groups by putting things near their nearest "center point"
# Unsupervised Learning: The model figures out patterns on its own (no labels needed)
np.set_printoptions(linewidth=200, threshold=1000) # Makes our arrays print nicely later on 

# ====== Step 1: Data Pre-Processing
# Grabbing dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Just for our reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(f"Training images: {x_train.shape}")
print(f"Each image is {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(x_train[0])
plt.imshow(x_train[42], cmap='gray')
plt.title(class_names[y_train[42]])
plt.show()

# Flatten each 28x28 image into a 784-length array
x_train_flat = x_train.reshape(len(x_train), -1)

# Normalize to 0-1 range (helps clustering work better)
x_train_flat = x_train_flat / 255.0

# Use smaller sample for faster training
sample_size = 10000
x_sample = x_train_flat[:sample_size]
y_sample = y_train[:sample_size]

# ============= Step 2: Clustering =============
# K=10 because Fashion MNIST has 10 categories (but the model doesn't know this)
x_train_flat = x_train.reshape(len(x_train), -1)

# Normalize to 0-1 range (helps clustering work better)
x_train_flat = x_train_flat / 255.0

# Use smaller sample for faster training (you can increase this!)
sample_size = 10000
x_sample = x_train_flat[:sample_size]
y_sample = y_train[:sample_size]

# ============= Step 2: Clustering =============
# K=10 because Fashion MNIST has 10 categories (but the model doesn't know this!)
kmeans = sklearn.cluster.KMeans(n_clusters=10, random_state=42, n_init=10)
print("\nTraining K-Means clustering...")

print("\nTraining K-Means clustering...")
clusters = kmeans.fit_predict(x_sample)

print(f"Clustered {sample_size} images into 10 groups!")

# ============= Step 3: Visualize Results =============
# Show a few examples from each cluster
fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle('10 Random Samples from Each Cluster', fontsize=16)

for cluster_id in range(10):
    # Find all images in this cluster
    cluster_indices = np.where(clusters == cluster_id)[0]
    
    # Pick 10 random samples from this cluster
    if len(cluster_indices) >= 10:
        samples = np.random.choice(cluster_indices, 10, replace=False)
    else:
        samples = cluster_indices
    
    # Display them
    for i, idx in enumerate(samples):
        axes[cluster_id, i].imshow(x_train[idx], cmap='gray')
        axes[cluster_id, i].axis('off')
        
        # Label first image of each row with cluster number
        if i == 0:
            axes[cluster_id, i].set_title(f'Cluster {cluster_id}', fontsize=10)

plt.tight_layout()
plt.show()

# ============= Step 4: Analyze What Each Cluster Found =============
print("\n=== What did each cluster group together? ===")
for cluster_id in range(10):
    # Find which actual clothing types ended up in this cluster
    cluster_mask = clusters == cluster_id
    cluster_labels = y_sample[cluster_mask]
    
    # Count each type
    unique, counts = np.unique(cluster_labels, return_counts=True)
    
    print(f"\nCluster {cluster_id} ({len(cluster_labels)} items):")
    for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:3]:
        percentage = (count / len(cluster_labels)) * 100
        print(f"  {class_names[label]}: {count} ({percentage:.1f}%)")

# ============= Predict cluster for new image =============
# print("\n=== Testing on a new image ===")
# test_image = x_test[0].reshape(1, -1) / 255.0
# predicted_cluster = kmeans.predict(test_image)[0]
# actual_label = class_names[y_test[0]]

# print(f"This {actual_label} was assigned to Cluster {predicted_cluster}")
# plt.imshow(x_test[0], cmap='gray')
# plt.title(f'Test Image: {actual_label} → Cluster {predicted_cluster}')
# plt.axis('off')
# plt.show()