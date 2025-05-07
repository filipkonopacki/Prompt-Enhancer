import os
import random

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from utils.definitions import tokenized_dataset_t5_path, tokenized_dataset_mistral_path, instruction_templates
import matplotlib.pyplot as plt

# Models
t5 = "google/flan-t5-base"
mistral = "mistralai/Mistral-7B-v0.3"

# Dataset
prompt_enhancer = "gokaygokay/prompt-enhancer-dataset"


casual_llm = "casual_llm"
seq2seq = "seq2seq"
model_type = casual_llm
tokenized_dataset_path = tokenized_dataset_t5_path if model_type == seq2seq else tokenized_dataset_mistral_path

if model_type == seq2seq:
    tokenizer = AutoTokenizer.from_pretrained(t5, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(mistral, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token


# Format dataset for Seq2Seq
def format_data(examples):
    inputs = [f"{random.choice(instruction_templates)}\n{short_prompt}" for short_prompt in examples['short_prompt']]
    outputs = [long_prompt for long_prompt in examples['long_prompt']]

    return {
        "input": inputs,
        "output": outputs
    }


# Format dataset for CasualLLM
def format_data_casual_llm(examples):
    inputs = [f"{random.choice(instruction_templates)}\n{short_prompt}" for short_prompt in examples['short_prompt']]
    outputs = [long_prompt for long_prompt in examples['long_prompt']]
    full_texts = [inp + "\n" + out for inp, out in zip(inputs, outputs)]

    return {"text": full_texts}


# Tokenize dataset for Seq2Seq
def tokenize_data(examples):
    inputs = tokenizer(examples["input"], padding="max_length", max_length=200, truncation=True)
    labels = tokenizer(examples["output"], padding="max_length", max_length=200, truncation=True)

    labels_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels_ids
    }


# Tokenize dataset for CasualLLM
def tokenize_data_casual_llm(examples):
    return tokenizer(examples["text"], truncation=False, padding=False)


def format_save_dataset():
    # Load the dataset
    dataset = load_dataset(prompt_enhancer)
    if model_type == seq2seq:
        # Format dataset to contain input (enhance instruction + short prompt) and output (long prompt)
        formatted_dataset = dataset.map(format_data, batched=True)
        # Tokenize dataset
        tokenized_dataset = formatted_dataset.map(tokenize_data, batched=True)
        # Remove unnecessary columns, leave only input_ids, attention_mask, and labels
        tokenized_dataset = tokenized_dataset.remove_columns(["input", "output", "short_prompt", "long_prompt"])
        tokenized_dataset.save_to_disk(tokenized_dataset_path)
    else:
        # Format dataset to contain full text (enhance instruction + short prompt + long prompt)
        formatted_dataset = dataset.map(format_data_casual_llm, batched=True)
        # Tokenize dataset
        tokenized_dataset = formatted_dataset.map(tokenize_data_casual_llm, batched=True)
        # Remove unnecessary columns, leave only input_ids, attention_mask, and labels
        tokenized_dataset = tokenized_dataset.remove_columns(["text", "short_prompt", "long_prompt"])
        tokenized_dataset.save_to_disk(tokenized_dataset_path)

    return tokenized_dataset


def get_dataset():
    if not os.path.exists(tokenized_dataset_path):
        tokenized_dataset = format_save_dataset()
    else:
        tokenized_dataset = load_from_disk(tokenized_dataset_path)

    return tokenized_dataset


if __name__ == "__main__":
    # Testing tokenization
    dataset = load_dataset(prompt_enhancer)
    formatted_dataset = dataset.map(format_data_casual_llm, batched=True)
    tokenized_dataset = formatted_dataset.map(tokenize_data_casual_llm, batched=True)
    lengths = [len(x) for x in tokenized_dataset["train"]["input_ids"]]
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {sum(lengths) / len(lengths):.2f}")
    print(f"Number of entries at max length (512): {sum(l == 512 for l in lengths)}")
    print(tokenized_dataset['train'][0]['input_ids'])
    print(tokenizer.decode(tokenized_dataset['train'][20]['input_ids']))


    # Plot the distribution of token lengths
    plt.hist(lengths, bins=50)
    plt.title("Tokenized Input Lengths Distribution")
    plt.xlabel("Token Length")
    plt.ylabel("Number of Samples")
    plt.show()
