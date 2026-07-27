import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Input documents
docs = []

n = int(input("Enter number of documents: "))

for i in range(n):
    docs.append(input("Enter document: "))

# Input search query
query = input("\nEnter search query: ")

# Convert documents into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Convert query into TF-IDF vector
query_vec = vectorizer.transform([query])

# Calculate TF-IDF cosine similarity
scores = cosine_similarity(query_vec, X)

print("\nTF-IDF Similarity Scores:")
for i, s in enumerate(scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Apply Latent Semantic Analysis (LSA)
svd = TruncatedSVD(n_components=2)
X_lsa = svd.fit_transform(X)
query_lsa = svd.transform(query_vec)

# Calculate LSA cosine similarity
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\nLSA Similarity Scores:")
for i, s in enumerate(lsa_scores[0]):
    print("Document", i + 1, ":", round(s, 3))

# Find the most relevant document
best = np.argmax(lsa_scores)

print("\nMost Relevant Document:")
print("Document", best + 1, ":", docs[best])