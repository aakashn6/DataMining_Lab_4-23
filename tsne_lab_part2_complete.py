import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

# Step 1: Load the digits dataset
digits = load_digits()
X = digits.data[:100]   # Use only 100 samples for faster t-SNE execution
y = digits.target[:100]

print("Shape of X:", X.shape)  # (100, 64)
print("Each sample has 64 features representing an 8x8 image of a handwritten digit.")

# Step 2: Run t-SNE with different perplexity values
perplexities = [5, 30, 50]

for perplexity in perplexities:
    print(f"Running t-SNE with perplexity = {perplexity}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X)

    # Step 3: Visualize the 2D t-SNE embedding
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='tab10', s=30)
    plt.title(f"t-SNE Embedding (Perplexity = {perplexity})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    
    # Step 4: Save each figure
    plt.savefig(f"tsne_digits_perplexity_{perplexity}.png")
    plt.show()
