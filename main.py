import fasttext

# Path to the training data
training_data_path = 'output.txt'

# Train the FastText model
model = fasttext.train_supervised(input=training_data_path, epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1)

# Save the model for later use
model.save_model("name_non_name_model.bin")

# Evaluate the model on the same training data (or you can split your data into training and test sets)
result = model.test(training_data_path)

# Print evaluation results
print(f"Number of examples: {result[0]}")
print(f"Precision: {result[1]}")
print(f"Recall: {result[2]}")
