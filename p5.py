import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Input legal text
text = input("Enter legal text: ")

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

# Identify Proper Nouns (NNP) as Named Entities
for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

# Actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Calculate accuracy
if max(count, actual) == 0:
    accuracy = 100
else:
    accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")