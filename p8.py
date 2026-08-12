import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

# Input reviews
reviews = []

try:
    n = int(input("Enter number of reviews: "))
except ValueError:
    print("Please enter a valid number.")
    exit()

if n < 2:
    print("Please enter at least 2 reviews.")
    exit()

for i in range(n):
    review = input(f"Enter review {i + 1}: ").strip()
    reviews.append(review)

# Convert text into numerical form
vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(reviews)

# Topic Modeling using LDA
lda = LatentDirichletAllocation(
    n_components=2,
    random_state=42
)
lda.fit(X)

words = vectorizer.get_feature_names_out()

print("\nTopics Found:")

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i + 1}:")
    top_indices = topic.argsort()[-5:][::-1]
    for idx in top_indices:
        print(words[idx])

# Convert sparse matrix to dense
X_dense = X.toarray()

# Set perplexity automatically
perplexity = min(5, len(reviews) - 1)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity
)

X_tsne = tsne.fit_transform(X_dense)

print("\nt-SNE Coordinates:")
for i, point in enumerate(X_tsne):
    print(f"Review {i + 1}: {point}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=100)

for i in range(len(reviews)):
    plt.text(
        X_tsne[i, 0],
        X_tsne[i, 1],
        f"R{i+1}",
        fontsize=10
    )

plt.title("t-SNE Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
