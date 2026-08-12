from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Lists to store documents and labels
docs = []
labels = []

# Get number of documents
n = int(input("Enter number of documents: "))

# Input documents and categories
for i in range(n):
    docs.append(input(f"Enter document {i + 1}: "))
    labels.append(input(f"Enter category {i + 1}: "))

# -------------------------------
# Rule-Based Classification
# -------------------------------

rule_pred = []

for doc in docs:
    doc = doc.lower()

    if "contract" in doc:
        rule_pred.append("contract")
    elif "judgment" in doc:
        rule_pred.append("judgment")
    else:
        rule_pred.append("agreement")

# Calculate Rule-Based Accuracy
rule_acc = accuracy_score(labels, rule_pred)

# -------------------------------
# Maximum Entropy Classification
# -------------------------------

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(docs)

# Create Maximum Entropy classifier
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X, labels)

# Predict categories
ml_pred = model.predict(X)

# Calculate accuracy
ml_acc = accuracy_score(labels, ml_pred)

# -------------------------------
# Display Results
# -------------------------------

print("\nRule-Based Predictions:")
for i, prediction in enumerate(rule_pred):
    print(f"Document {i + 1}: {prediction}")

print("\nMaximum Entropy Predictions:")
for i, prediction in enumerate(ml_pred):
    print(f"Document {i + 1}: {prediction}")

print("\nRule-Based Accuracy:", rule_acc)
print("Maximum Entropy Accuracy:", ml_acc)
