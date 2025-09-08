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
        img = cv2.imread(f'train/Dog/{f}', 0)  # grayscale
        if img is not None:
            dogs.append(cv2.resize(img, (8, 8)).flatten() / 255.0)

cats, dogs = np.array(cats), np.array(dogs)

print(f"Loaded {len(cats)} cat images and {len(dogs)} dog images for training.")

# -------------------------------
# 2. Compute statistics per class
# -------------------------------
cat_mean, cat_std = np.mean(cats, axis=0), np.std(cats, axis=0) + 1e-6
dog_mean, dog_std = np.mean(dogs, axis=0), np.std(dogs, axis=0) + 1e-6

# -------------------------------
# 3. Gaussian PDF
# -------------------------------
def gaussian_prob(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

# -------------------------------
# 4. Priors (equal)
# -------------------------------
prior_cat = prior_dog = 0.5

# -------------------------------
# 5. Test on all test images
# -------------------------------
y_true, y_pred = [], []

# Assuming test folder has subfolders: test/Cat and test/Dog
for label, folder in enumerate(["Cat", "Dog"]):  # 0=Cat, 1=Dog
    for f in os.listdir(f'test/{folder}'):
        if f.endswith('.jpg'):
            img = cv2.imread(f'test/{folder}/{f}', 0)
            if img is not None:
                test_img = cv2.resize(img, (8, 8)).flatten() / 255.0

                # Likelihoods
                cat_likelihood = np.prod(gaussian_prob(test_img, cat_mean, cat_std))
                dog_likelihood = np.prod(gaussian_prob(test_img, dog_mean, dog_std))

                # Posteriors
                posterior_cat = cat_likelihood * prior_cat
                posterior_dog = dog_likelihood * prior_dog
                total = posterior_cat + posterior_dog
                prob_cat = posterior_cat / total
                prob_dog = posterior_dog / total

                prediction = 0 if prob_cat > prob_dog else 1  # 0=Cat, 1=Dog

                y_true.append(label)
                y_pred.append(prediction)

# -------------------------------
# 6. Accuracy
# -------------------------------
y_true, y_pred = np.array(y_true), np.array(y_pred)
accuracy = np.mean(y_true == y_pred)

print("\n--- Test Results ---")
print(f"Total Test Images: {len(y_true)}")
print(f"Correct Predictions: {(y_true == y_pred).sum()}")
print(f"Accuracy: {accuracy:.3f}")

# -------------------------------
# 7. External Image Prediction
# -------------------------------
def classify_image(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
    test_img = cv2.resize(img, (8, 8)).flatten() / 255.0

    # Likelihoods
    cat_likelihood = np.prod(gaussian_prob(test_img, cat_mean, cat_std))
    dog_likelihood = np.prod(gaussian_prob(test_img, dog_mean, dog_std))

    # Posteriors
    posterior_cat = cat_likelihood * prior_cat
    posterior_dog = dog_likelihood * prior_dog
    total = posterior_cat + posterior_dog
    prob_cat = posterior_cat / total
    prob_dog = posterior_dog / total

    prediction = "Cat" if prob_cat > prob_dog else "Dog"

    print("\n--- External Image Prediction ---")
    print(f"Image: {img_path}")
    print(f"Posterior Cat: {prob_cat:.3f}")
    print(f"Posterior Dog: {prob_dog:.3f}")
    print(f"Prediction: {prediction}")

# Example usage:
classify_image("99c.jpg")
