import fasttext

# Load the trained model
model = fasttext.load_model("name_non_name_model.bin")

# Define a list of words to test
test_words = [
    "robert olaf magnus",   # Example of a name
    "Chair",    # Example of a non-name
    "John",     # Example of a name
    "furniture" # Example of a non-name
]

# Predict the labels for each test word
for word in test_words:
    labels, probabilities = model.predict(word)
    print(f"Word: {word}")
    print(f"Predicted Label: {labels[0]}")
    print(f"Probability: {probabilities[0]}")
    print()
