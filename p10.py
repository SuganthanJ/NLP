from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Get posts from the user
posts = []

n = int(input("Enter number of posts: "))

for i in range(n):
    post = input("Enter post: ")
    posts.append(post)

# Get number of clusters
k = int(input("Enter number of clusters: "))

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(posts)

# Create and train K-Means model
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

model.fit(X)

# Get cluster labels
labels = model.labels_

# Display cluster results
print("\nCluster Results:\n")

for i in range(len(posts)):
    print("Post:", posts[i])
    print("Cluster:", labels[i])
    print()

# Get important keywords
terms = vectorizer.get_feature_names_out()

print("Important Keywords:\n")

for i in range(k):
    center = model.cluster_centers_[i]

    # Get indices of top 5 keywords
    top = center.argsort()[-5:][::-1]

    print("Cluster", i)

    for j in top:
        print(terms[j])

    print()

# Marketing insight
print("Marketing Insight:")
print("Similar customer opinions are grouped together.")
print("Clusters help identify product trends and issues.")
