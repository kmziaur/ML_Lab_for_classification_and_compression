import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Load training data
# -------------------------------
cats, dogs = [], []

for f in os.listdir('train/Cat'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Cat/{f}', 0)  # grayscale
        if img is not None:
            img = cv2.resize(img, (16, 16))  # 16x16 = 256 features
            cats.append(img.flatten() / 255.0)  # normalize

for f in os.listdir('train/Dog'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Dog/{f}', 0)
        if img is not None:
            img = cv2.resize(img, (16, 16))
            dogs.append(img.flatten() / 255.0)

cats, dogs = np.array(cats), np.array(dogs)

print(f"Loaded {len(cats)} cats and {len(dogs)} dogs for training.")

X_train = np.vstack((cats, dogs))
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=Cat, 1=Dog

# -------------------------------
# 2. Manual LDA Training
# -------------------------------
# Compute means
mean_cat = np.mean(cats, axis=0)
mean_dog = np.mean(dogs, axis=0)

# Within-class scatter matrix
Sw = np.cov(cats.T) * (len(cats)-1) + np.cov(dogs.T) * (len(dogs)-1)

# Between-class scatter
mean_diff = (mean_cat - mean_dog).reshape(-1, 1)
Sb = np.dot(mean_diff, mean_diff.T)

# Solve generalized eigenvalue problem: inv(Sw) * (mean_cat - mean_dog)
Sw_inv = np.linalg.pinv(Sw)  # pseudo-inverse for stability
w = np.dot(Sw_inv, (mean_cat - mean_dog))  # projection vector

# Normalize projection
w = w / np.linalg.norm(w)

# Project training data onto w
proj_cats = np.dot(cats, w)
proj_dogs = np.dot(dogs, w)

# Compute class thresholds (midpoint of means in projected space)
mean_proj_cat = np.mean(proj_cats)
mean_proj_dog = np.mean(proj_dogs)
threshold = (mean_proj_cat + mean_proj_dog) / 2

# -------------------------------
# 3. Prediction function
# -------------------------------
def lda_predict(x):
    """Predict 0=Cat, 1=Dog for a given image vector x."""
    proj = np.dot(x, w)
    return 0 if abs(proj - mean_proj_cat) < abs(proj - mean_proj_dog) else 1

# -------------------------------
# 4. Test on all test images
# -------------------------------
y_true, y_pred = [], []

for label, folder in enumerate(["Cat", "Dog"]):  # 0=Cat, 1=Dog
    for f in os.listdir(f'test/{folder}'):
        if f.endswith('.jpg'):
            img = cv2.imread(f'test/{folder}/{f}', 0)
            if img is not None:
                img = cv2.resize(img, (16, 16))
                x_test = img.flatten() / 255.0

                prediction = lda_predict(x_test)
                y_true.append(label)
                y_pred.append(prediction)

# Accuracy
y_true, y_pred = np.array(y_true), np.array(y_pred)
accuracy = accuracy_score(y_true, y_pred)

print("\n--- Test Results ---")
print(f"Total Test Images: {len(y_true)}")
print(f"Correct Predictions: {(y_true == y_pred).sum()}")
print(f"Accuracy: {accuracy:.3f}")

# -------------------------------
# 5. External Image Prediction
# -------------------------------
def classify_image(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    img = cv2.resize(img, (16, 16))
    x_test = img.flatten() / 255.0

    prediction = lda_predict(x_test)
    result = "Cat" if prediction == 0 else "Dog"

    print("\n--- External Image Prediction ---")
    print(f"Image: {img_path}")
    print(f"Prediction: {result}")

# Example usage:
classify_image("99d.jpg")
