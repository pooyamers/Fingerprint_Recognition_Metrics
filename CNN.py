import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Parameters
data_dir = "C:/Users/4dmer/Desktop/Projects/Fingerprints/DB1_B"
img_size = (128, 128)
batch_size = 32
epochs = 10

# Load dataset
class_names = sorted(os.listdir(data_dir))
print("Class names:", class_names)

file_paths = []
labels = []

for label_idx, class_name in enumerate(class_names):
    class_folder = os.path.join(data_dir, class_name)
    for file_name in os.listdir(class_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_paths.append(os.path.join(class_folder, file_name))
            labels.append(label_idx)

file_paths = np.array(file_paths)
labels = np.array(labels)
print(f"Total images: {len(file_paths)}")


def load_and_preprocess_image(path):
    """
    Helper function to load and preprocess a single image.
    Args:
        path (str): Path to the image file.
    Returns:
        tf.Tensor: Preprocessed image tensor; s.t. shape is (img_size[0], img_size[1], 3), 
        pixel values are normalized to [0, 1].
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image

def make_pairs(file_paths, labels):
    """
    Helper function to generate pairs of images for Siamese training.
    Args:
        file_paths (list): List of image file paths.
        labels (list): Corresponding list of labels for the images.
    Returns:
        np.array: Array of pairs of image file paths.
        np.array: Array of labels for the pairs (1 for positive, 0 for negative). 
    """
    pairs = []
    pair_labels = []
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)

    for idx_a in range(len(file_paths)):
        img_a = file_paths[idx_a]
        label_a = labels[idx_a]

        # Positive pair
        idx_b = idx_a
        while idx_b == idx_a:
            idx_b = random.choice(label_to_indices[label_a])
        img_b = file_paths[idx_b]
        pairs.append([img_a, img_b])
        pair_labels.append(1)

        # Negative pair
        neg_label = label_a
        while neg_label == label_a:
            neg_label = random.choice(list(label_to_indices.keys()))
        idx_b = random.choice(label_to_indices[neg_label])
        img_b = file_paths[idx_b]
        pairs.append([img_a, img_b])
        pair_labels.append(0)

    return np.array(pairs), np.array(pair_labels)

pairs, pair_labels = make_pairs(file_paths, labels)
print(f"Total pairs: {len(pairs)}")

def preprocess_pair(pair, label):
    """
    Helper function to preprocess a pair of images.
    Args:
        pair (list): List containing two image file paths.
        label (int): Label for the pair (1 for positive, 0 for negative).
    Returns:
        tuple: Tuple containing two preprocessed image tensors and the label.
    """
    img1 = load_and_preprocess_image(pair[0])
    img2 = load_and_preprocess_image(pair[1])
    return (img1, img2), label

# Split pairs into training and validation sets with stratification sampling
pairs_train, pairs_val, labels_train, labels_val = train_test_split(
    pairs, pair_labels, test_size=0.2, random_state=42, stratify=pair_labels
)

def create_dataset(pairs, labels, batch_size=32, shuffle=True):
    """
    Helper function to create a tf.data.Dataset pipeline from pairs of images and labels.
    Args:
        pairs (np.array): Array of pairs of image file paths.
        labels (np.array): Array of labels for the pairs.
        batch_size (int): Size of the batches(a power of 2); default is 32.
        shuffle (bool): Whether to shuffle the dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((pairs, labels))
    dataset = dataset.map(preprocess_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Build datasets for training and validation
train_dataset = create_dataset(pairs_train, labels_train, batch_size=batch_size, shuffle=True)
val_dataset = create_dataset(pairs_val, labels_val, batch_size=batch_size, shuffle=False)

def build_base_model(input_shape=img_size + (3,)):
    """
    Transfer learning:
    Helper function to build the base CNN model on top of MobileNetV2 for feature extraction.
    Args:
        input_shape (tuple): Shape of the input images; default is (128, 128, 3).
    Returns:
        Model: Keras Model object representing the base CNN.
    """
    base_cnn = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_cnn.trainable = False
    inputs = layers.Input(shape=input_shape)
    x = base_cnn(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
    return Model(inputs, x)

base_model = build_base_model()

# Define inputs for the two images in a pair
input_a = layers.Input(shape=img_size + (3,))
input_b = layers.Input(shape=img_size + (3,))

# Generate embeddings for both images using the base model
embedding_a = base_model(input_a)
embedding_b = base_model(input_b)

# Calculate absolute difference (L1 distance) between embeddings
l1_distance = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([embedding_a, embedding_b])

# Final dense layer outputs similarity score between 0 and 1
output = layers.Dense(1, activation='sigmoid')(l1_distance)

# Define Siamese model taking two images and outputting similarity score
siamese_model = Model(inputs=[input_a, input_b], outputs=output)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with validation
history = siamese_model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- Recognition test ---

# Function to get embeddings from base model
def get_embeddings(file_paths):
    images = [load_and_preprocess_image(p) for p in file_paths]
    images = tf.stack(images)
    embeddings = base_model.predict(images, batch_size=batch_size)
    return embeddings

# Get embeddings for all images
embeddings = get_embeddings(file_paths)

# Recognition by Nearest Neighbor
# For each image, find the closest other embedding (excluding itself) and check if labels match

correct = 0
total = len(file_paths)

# Compute pairwise distances (Euclidean)
dist_matrix = cdist(embeddings, embeddings, metric='euclidean')

for i in range(total):
    # Set self-distance to large number to exclude
    dist_matrix[i, i] = np.inf
    nearest_idx = np.argmin(dist_matrix[i])
    if labels[i] == labels[nearest_idx]:
        correct += 1

recognition_accuracy = correct / total
print(f"Recognition accuracy by nearest neighbor: {recognition_accuracy:.4f}")

# Final training accuracy: 0.9865
# Final validation accuracy: 0.5625
# Recognition accuracy by nearest neighbor: 0.7125