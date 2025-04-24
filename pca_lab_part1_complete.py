import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a 2D dataset with specified mean and covariance
np.random.seed(42)
X = np.random.multivariate_normal(mean=[2, 5], cov=[[3, 2], [2, 2]], size=100)

# Step 2: Plot the original 2D data
plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
plt.title("Original 2D Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.grid(True)
plt.savefig("pca_original_data.png")
plt.show()

# Step 3: Center the data (subtract mean from each feature)
X_centered = X - np.mean(X, axis=0)

# Step 4: Compute the covariance matrix
cov_matrix = np.cov(X_centered.T)
print("Covariance Matrix:\n", cov_matrix)

# Step 5: Compute eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
print("Eigenvalues:", eig_vals)
print("Eigenvectors:\n", eig_vecs)

# Step 6: Project the data onto the first principal component (PC1)
pc1 = eig_vecs[:, 0]
X_pca_1d = X_centered @ pc1  # Matrix multiplication to get 1D projection

# Step 7: Plot the 1D projection
plt.scatter(X_pca_1d, np.zeros_like(X_pca_1d), alpha=0.7)
plt.title("1D Projection onto First Principal Component")
plt.xlabel("PC1")
plt.grid(True)
plt.savefig("pca_1d_projection.png")
plt.show()

# Step 8: Plot the variance explained by each component
explained_variance = eig_vals / np.sum(eig_vals)
plt.bar(["PC1", "PC2"], explained_variance)
plt.title("Explained Variance by Principal Components")
plt.ylabel("Proportion of Variance Explained")
plt.grid(True)
plt.savefig("pca_explained_variance.png")
plt.show()
