import os
import cv2
import numpy as np

# -------------------------------
# 1. Load training data
# -------------------------------
cats, dogs = [], []

for f in os.listdir('train/Cat'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Cat/{f}', 0)
        if img is not None:
            cats.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

for f in os.listdir('train/Dog'):
    if f.endswith('.jpg'):
        img = cv2.imread(f'train/Dog/{f}', 0)
        if img is not None:
            dogs.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

X_train = np.vstack((cats, dogs))
y_train = np.array([0]*len(cats) + [1]*len(dogs))  # 0=Cat, 1=Dog

print(f"Loaded {len(cats)} cat images and {len(dogs)} dog images for training.")

# -------------------------------
# 2. Add bias term
# -------------------------------
X_train_bias = np.column_stack([np.ones(len(X_train)), X_train])

# -------------------------------
# 3. Sigmoid and gradient descent
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.5, epochs=500):
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ w
        y_pred = sigmoid(z)
        grad = X.T @ (y_pred - y) / len(y)
        w -= lr * grad
    return w

weights = train_logistic_regression(X_train_bias, y_train)
print("Training completed.")

# -------------------------------
# 4. Test on all test images
# -------------------------------
y_true, y_pred = [], []

for label, folder in enumerate(["Cat", "Dog"]):
    for f in os.listdir(f'test/{folder}'):
        if f.endswith('.jpg'):
            img = cv2.imread(f'test/{folder}/{f}', 0)
            if img is not None:
                x_test = cv2.resize(img, (8, 8)).flatten() / 255.0
                x_test_bias = np.concatenate([[1], x_test])
                prob = sigmoid(x_test_bias @ weights)
                prediction = 0 if prob < 0.5 else 1
                y_true.append(label)
                y_pred.append(prediction)

y_true, y_pred = np.array(y_true), np.array(y_pred)
accuracy = np.mean(y_true == y_pred)

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
    x_test = cv2.resize(img, (8, 8)).flatten() / 255.0
    x_test_bias = np.concatenate([[1], x_test])
    prob = sigmoid(x_test_bias @ weights)
    prediction = "Cat" if prob < 0.5 else "Dog"
    print("\n--- External Image Prediction ---")
    print(f"Image: {img_path}")
    print(f"Probability Cat: {1 - prob:.3f}")
    print(f"Probability Dog: {prob:.3f}")
    print(f"Prediction: {prediction}")

# Example usage
classify_image("99c.jpg")
