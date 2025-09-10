from datasets import load_dataset

dataset = load_dataset("text", data_files="./tinyshakespeare.txt")
print("dataset created successfully")

split_dataset = dataset["train"].train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]
print(train_dataset)
print(test_dataset)
