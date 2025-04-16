import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer

from definitions import output_dir, logging_dir, final_lora, timestamp
from data.format_dataset import get_dataset
from evaluate import load

# Models
model_name = "google/flan-t5-base"

rouge = load("rouge")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Decode generated predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as they're not going to be decoded correctly
    labels = labels.copy()  # Create a copy to avoid modifying the original
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    # Log a few examples for inspection
    if len(decoded_preds) > 2:
        print("\nExample predictions:")
        for i in range(2):
            print(f"Original: {decoded_labels[i][:100]}...")
            print(f"Enhanced: {decoded_preds[i][:100]}...")
            print("---")

    return {
        f"rouge_{k}": v.mid.fmeasure
        for k, v in result.items()
    }


lora_config = LoraConfig(
    r=8,  # Rank of LoRA matrices (tradeoff: higher = more expressive, lower = faster/smaller)
    lora_alpha=16,  # Scaling factor (common heuristic: alpha = 2 * r)
    target_modules=["q", "v"],  # T5 uses "q", "v" for attention projections
    lora_dropout=0.1,
    bias="none",  # Don't adapt bias terms
    task_type=TaskType.SEQ_2_SEQ_LM  # Important! This sets it up for T5-style generation
)

# Load and prepare dataset
tokenized_dataset = get_dataset()
# Print dataset statistics
print(f"Training examples: {len(tokenized_dataset['train'])}")
print(f"Test examples: {len(tokenized_dataset['test'])}")

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,                        # Where final model + checkpoints will be saved
    per_device_train_batch_size=4,                # Batch size per GPU (adjust for memory)
    per_device_eval_batch_size=4,                 # Batch size for eval
    gradient_accumulation_steps=8,                # Effective batch size = 8 x 4 = 32
    learning_rate=2e-4,                           # Learning rate — 2e-4 is good for LoRA
    num_train_epochs=3,                           # Number of full passes over training set
    logging_dir=logging_dir,                      # Where logs for TensorBoard or text go
    logging_steps=10,                             # Log every 50 steps
    # eval_strategy="steps",                        # Run eval at the end of every epoch
    # eval_steps=100,                               # Evaluate every 100 steps
    save_strategy="steps",                        # Save model checkpoint at end of every epoch
    save_steps=100,                               # Save every 100 steps
    save_total_limit=2,                           # Keep only last 2 checkpoints
    fp16=False,                                   # Mixed precision — faster training on GPUs but for T5 just doesn't work
    report_to="tensorboard",                      # Set to "wandb" or "tensorboard" if you want logging
    # load_best_model_at_end=True,                  # After training, keep the best eval-loss model
    metric_for_best_model="rouge1",               # Metric to use for best model selection
    # predict_with_generate=True,
    # generation_max_length=200,                    # Instead of max_new_tokens
    # generation_num_beams=4,                       # Beam search width
    run_name=f"lora-t5-prompt-enhancer--{timestamp}",   # Unique run name for logging
    label_names=["labels"],                       # Important! This is how we tell Trainer to use the labels
    warmup_ratio=0.1,                             # 10% of training steps for warmup
    max_grad_norm=1.0
)


peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding="max_length", max_length=200)


trainer = Seq2SeqTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
peft_model.save_pretrained(final_lora)
print(f"LoRA adapter saved to {final_lora}")

# Generate a few examples from test set
print("\nGenerating examples from test set:")
for i in range(3):
    example = tokenized_dataset["test"][i]
    input_text = tokenizer.decode(example["input_ids"], skip_special_tokens=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = peft_model.generate(
        input_ids=input_ids,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reference = tokenizer.decode(example["labels"], skip_special_tokens=True)

    print(f"Input: {input_text[:100]}...")
    print(f"Generated: {generated_text[:100]}...")
    print(f"Reference: {reference[:100]}...")
    print("---")
