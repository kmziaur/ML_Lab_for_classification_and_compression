import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# -------------------------------
# Helper: load images from folder
# -------------------------------
def load_images_from_folder(folder, label, size=(16, 16)):
    data, labels = [], []
    for f in os.listdir(folder):
        if f.endswith('.jpg'):
            path = os.path.join(folder, f)
            img = cv2.imread(path, 0)  # grayscale
            if img is not None:
                img = cv2.resize(img, size)
                data.append(img.flatten() / 255.0)  # normalize
                labels.append(label)
    return data, labels

# -------------------------------
# 1. Load training data
# -------------------------------
cats, y_cats = load_images_from_folder("train/Cat", 0)
dogs, y_dogs = load_images_from_folder("train/Dog", 1)

X_train = np.array(cats + dogs)
y_train = np.array(y_cats + y_dogs)

print(f"Training data shape: {X_train.shape}")
print(f"Cats: {len(cats)}, Dogs: {len(dogs)}")

# -------------------------------
# 2. Train SVM
# -------------------------------
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train, y_train)

# Training accuracy
train_acc = accuracy_score(y_train, svm.predict(X_train))
print(f"Training accuracy: {train_acc:.2f}")

# -------------------------------
# 3. Test on all test images
# -------------------------------
X_test, y_test = [], []
for label, folder in enumerate(["test/Cat", "test/Dog"]):
    data, labels = load_images_from_folder(folder, label)
    X_test.extend(data)
    y_test.extend(labels)

X_test, y_test = np.array(X_test), np.array(y_test)
y_pred = svm.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {test_acc:.2f}")

# -------------------------------
# 4. External image prediction
# -------------------------------
def predict_external(image_path, size=(16, 16)):
    img = cv2.imread(image_path, 0)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    img = cv2.resize(img, size).flatten() / 255.0
    pred = svm.predict([img])[0]
    conf = svm.decision_function([img])[0]
    result = "Cat" if pred == 0 else "Dog"
    print(f"\nExternal Image Prediction: {result}")
    print(f"Confidence: {abs(conf):.2f}")

# Example usage:
predict_external("99c.jpg")
