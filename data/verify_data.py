from data.format_dataset import get_dataset
from transformers import AutoTokenizer


def inspect_tokenized_dataset(num_samples=5):
    # Load the dataset
    dataset = get_dataset()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", use_fast=True)

    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        print(f"DatasetDict with splits: {list(dataset.keys())}")
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            first_split = list(dataset.keys())[0]
            dataset = dataset[first_split]

    print("Dataset loaded!")
    print(f"Number of samples: {len(dataset)}")
    print(f"Features: {dataset.features.keys()}\n")

    for i in range(min(num_samples, len(dataset))):
        example = dataset[i]
        input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)

        label_ids = [id for id in example["labels"] if id != -100]
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True)

        print(f"--- Sample {i} ---")
        print("Input:")
        print(input_text)
        print("\nLabel:")
        print(label_text)
        print("=" * 40)

    # Checks
    sample = dataset[0]
    assert "input_ids" in sample and "labels" in sample, "Missing keys!"
    assert isinstance(sample["labels"], list), "Labels should be a list of token IDs!"
    assert all(isinstance(i, int) for i in sample["input_ids"]), "Non-int tokens in input_ids!"
    assert all(isinstance(i, int) or i == -100 for i in sample["labels"]), "Invalid labels!"
    print("\n✅ Formatting checks passed!")

    return dataset


if __name__ == "__main__":
    inspect_tokenized_dataset()
