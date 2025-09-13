from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

load_data = load_dataset("text", data_files="./tinyshakespeare.txt")
print("dataset created successfully")

# first split train-test, train_val["test"] will be test
# second split train-val split, 80% of data, train_test["train"] will be train and train_test["test"] will be val
train_val = load_data["train"].train_test_split(
    test_size=0.2, seed=0
)  # 80% train 20% test
train_test = train_val["train"].train_test_split(
    test_size=0.125, seed=0
)  # validation 10%, 12.5% of train

dataset = {
    "train": train_test["train"],
    "validation": train_test["test"],
    "test": train_val["test"],
}

print(dataset)
print(f"training dataset is {dataset['train']}")

# import and create tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # required for gpt-2
print("tokenizer made")
print(tokenizer)


def tokenize_func(data):
    # longer than 512 gets truncated, shorter than that gets padded
    return tokenizer(data["text"], truncation=True, padding=True, max_length=512)


# take datasets and process in batches, call tokenize_func each batch, return new dataset
# pre-tokenizing each dataset for speed and memory efficiency
# for larger datasets that didn't fit into memory we'd tokenize on the fly
tokenized_train = dataset["train"].map(tokenize_func, batched=True)
tokenized_val = dataset["validation"].map(tokenize_func, batched=True)
tokenized_test = dataset["test"].map(tokenize_func, batched=True)

# features has text, input_ids and attention_mask now
print(f"tokenized train={tokenized_train}")
print(f"tokenized val={tokenized_val}")
print(f"tokenized test={tokenized_test}")

model = GPT2LMHeadModel.from_pretrained("gpt2")
print(f"model is {model}")

# will use collator for language modeling bc we're working with LLM
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # for causal language modeling like GPT
)

# if training issues change bf16 to fp16
train_params = TrainingArguments(
    do_train=True,
    output_dir="./model",  # saving checkpoints here
    bf16=True,
    gradient_accumulation_steps=2,
    num_train_epochs=3,  # low for fine tuning
    learning_rate=5e-5,  # IMPORTANT!! for fine tuning
    per_device_train_batch_size=2,  # low ram on my hardware
    eval_strategy="epoch",  # we can do steps if training takes longer, epoch cleaner rn
    warmup_steps=500,  # hit target lr this many steps after start
    save_only_model=True,
)

trainer = Trainer(
    model,
    args=train_params,
    data_collator=collator,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

trainer.train()
