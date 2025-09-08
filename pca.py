import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load all training images to learn PCA
# -------------------------------
def load_images(folder, size=(16,16)):
    data = []
    for f in os.listdir(folder):
        if f.endswith('.jpg'):
            img = cv2.imread(os.path.join(folder, f), 0)
            if img is not None:
                img = cv2.resize(img, size).flatten() / 255.0
                data.append(img)
    return np.array(data)

X_cats = load_images("train/Cat")
X_dogs = load_images("train/Dog")
X_train = np.vstack((X_cats, X_dogs))
print("Training images loaded:", X_train.shape)

# -------------------------------
# 2. Compute mean and center data
# -------------------------------
mean_vector = np.mean(X_train, axis=0)
X_centered = X_train - mean_vector

# -------------------------------
# 3. Compute covariance and eigen decomposition
# -------------------------------
cov_matrix = np.cov(X_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov_matrix)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# -------------------------------
# 4. Select top k components
# -------------------------------
k = 200
V_k = eigvecs[:, :k]

# -------------------------------
# 5. Function to reconstruct image
# -------------------------------
def reconstruct_image(feature_vector):
    reconstructed = (feature_vector @ V_k.T) + mean_vector
    return reconstructed.reshape((16,16))

# -------------------------------
# 6. Load test/external image
# -------------------------------
test_path = "99d.jpg"  # replace with your test image path
img = cv2.imread(test_path, 0)
if img is None:
    raise FileNotFoundError(f"Could not load image: {test_path}")

img_flat = cv2.resize(img, (16,16)).flatten() / 255.0
img_centered = img_flat - mean_vector

# Project to PCA space
img_reduced = img_centered @ V_k
print("Reduced feature vector shape:", img_reduced.shape)

# Reconstruct image
img_reconstructed = reconstruct_image(img_reduced)

# Save reconstructed image
os.makedirs("reconstructed_test", exist_ok=True)
out_path = "reconstructed_test/reconstructed_0.png"
plt.imshow(img_reconstructed, cmap='gray')
plt.axis('off')
plt.title("Reconstructed Test Image")
plt.savefig(out_path)
plt.close()
print(f"Reconstructed test image saved to {out_path}")
