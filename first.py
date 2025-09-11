from datasets import load_dataset
from transformers import GPT2Tokenizer

load_data = load_dataset("text", data_files="./tinyshakespeare.txt")
print("dataset created successfully")

# first split train-test, train_val["test"] will be test
# second split train-val split, 80% of data, train_test["train"] will be train and train_test["test"] will be val
train_val = load_data["train"].train_test_split(test_size=0.2, seed=0) # 80% train 20% test
train_test = train_val["train"].train_test_split(test_size=0.125, seed=0) # validation 10%, 12.5% of train

dataset = {
    "train": train_test["train"],
    "validation": train_test["test"],
    "test": train_val["test"]
}

print(dataset)
print(f"training dataset is {dataset["train"]}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token # required for gpt-2
print("tokenizer made")
print(tokenizer)

def tokenize_func(data):
   # longer than 512 gets truncated, shorter than that gets padded
    return tokenizer(data["text"], truncation=True, padding=True, max_length=512)

# take datasets and process in batches, call tokenize_func each batch, return new dataset
tokenized_train = dataset["train"].map(tokenize_func, batched=True)
tokenized_val = dataset["validation"].map(tokenize_func, batched=True)
tokenized_test = dataset["test"].map(tokenize_func, batched=True)

print(tokenized_train)
