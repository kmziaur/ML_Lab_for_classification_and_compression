import os
import cv2
import numpy as np

# -------------------------------
# 1. Load training data
# -------------------------------
cats, dogs = [], []

for f in os.listdir('train/Cat'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Cat/{f}', 0)  # grayscale
        if img is not None:
            cats.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

for f in os.listdir('train/Dog'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Dog/{f}', 0)
        if img is not None:
            dogs.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

# Create training matrix and labels
X_train = np.array(cats + dogs)  # Training features
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=Cat, 1=Dog

# Add bias term (column of ones)
X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])

# -------------------------------
# 2. Least Squares Solution (Normal Equation)
# -------------------------------
XTX = X_train_bias.T @ X_train_bias
XTy = X_train_bias.T @ y_train
weights = np.linalg.solve(XTX, XTy)

# -------------------------------
# 3. Prediction function
# -------------------------------
def predict_image(img):
    """Classify an image (numpy array, grayscale)."""
    img = cv2.resize(img, (8, 8)).flatten() / 255.0
    x_bias = np.concatenate([[1], img])  # add bias term
    score = x_bias @ weights
    prob = 1 / (1 + np.exp(-score))  # sigmoid probability
    return (0 if prob < 0.5 else 1), prob  # return class & prob

# -------------------------------
# 4. Test on all test images
# -------------------------------
y_true, y_pred = [], []

for label, folder in enumerate(["Cat", "Dog"]):  # 0=Cat, 1=Dog
    for f in os.listdir(f'test/{folder}'):
        if f.endswith('.jpg'):
            img = cv2.imread(f'test/{folder}/{f}', 0)
            if img is not None:
                pred, prob = predict_image(img)
                y_true.append(label)
                y_pred.append(pred)

# Test accuracy
y_true, y_pred = np.array(y_true), np.array(y_pred)
test_accuracy = np.mean(y_true == y_pred)

print("\n--- Test Results ---")
print(f"Total Test Images: {len(y_true)}")
print(f"Correct Predictions: {(y_true == y_pred).sum()}")
print(f"Test Accuracy: {test_accuracy:.3f}")

# -------------------------------
# 5. External Image Prediction
# -------------------------------
def classify_image(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    pred, prob = predict_image(img)
    result = "Cat" if pred == 0 else "Dog"

    print("\n--- External Image Prediction ---")
    print(f"Image: {img_path}")
    print(f"Cat probability: {1 - prob:.3f}")
    print(f"Dog probability: {prob:.3f}")
    print(f"Prediction: {result}")

# Example usage:
classify_image("99d.jpg")
