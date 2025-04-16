import os
import random

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from utils.definitions import tokenized_dataset_path, instruction_templates

# Models
t5 = "google/flan-t5-base"
# Dataset
prompt_enhancer = "gokaygokay/prompt-enhancer-dataset"

tokenizer = AutoTokenizer.from_pretrained(t5, use_fast=True)


def format_data(examples):
    # Format inputs and outputs as lists
    instruction = random.choice(instruction_templates)
    inputs = [f"{instruction}:\n{short_prompt}" for short_prompt in examples['short_prompt']]
    outputs = [long_prompt for long_prompt in examples['long_prompt']]

    return {
        "input": inputs,
        "output": outputs
    }


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


def format_save_dataset():
    # Load the dataset
    dataset = load_dataset(prompt_enhancer)
    # Format dataset to contain input (enhance instruction + short prompt) and output (long prompt)
    formatted_dataset = dataset.map(format_data, batched=True)
    # Tokenize dataset
    tokenized_dataset = formatted_dataset.map(tokenize_data, batched=True)

    # Remove unnecessary columns, leave only input_ids, attention_mask, and labels
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output", "short_prompt", "long_prompt"])
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
    tokenized_dataset_test = get_dataset()
    print(tokenized_dataset_test["train"][0]['labels'])
