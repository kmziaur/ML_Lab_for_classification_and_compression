import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def manual_svd(A):
    """
    Compute SVD of matrix A manually: A = U * S * Vt
    """
    # Step 1: Compute A^T A
    ATA = A.T @ A

    # Step 2: Eigen decomposition of A^T A
    eigvals, V = np.linalg.eigh(ATA)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # Step 3: Compute singular values safely
    S = np.sqrt(np.maximum(eigvals, 0))  # clamp negative eigenvalues to 0

    # Step 4: Compute U = A V S^-1
    U = np.zeros((A.shape[0], len(S)))
    for i in range(len(S)):
        if S[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / S[i]
        else:
            U[:, i] = 0

    Vt = V.T
    return U, S, Vt

if __name__ == "__main__":
    # Load grayscale image
    img_path = "99c.jpg"  # Replace with your image path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(float)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")

    # Compute manual SVD
    U, S, Vt = manual_svd(img)
    print("Manual SVD completed. Shapes -> U:", U.shape, "S:", S.shape, "Vt:", Vt.shape)

    # Compress and reconstruct image with different k
    for k in [5, 20, 50]:
        approx = (U[:, :k] * S[:k]) @ Vt[:k, :]
        plt.imshow(approx, cmap="gray")
        plt.title(f"SVD Compression with k={k}")
        plt.axis('off')

        # Save reconstructed image
        out_path = f"svd_k{k}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")
