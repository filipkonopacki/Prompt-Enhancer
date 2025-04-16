import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
from peft import PeftModel

base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda")
model = PeftModel.from_pretrained(
    base_model,
    r'E:\Programming\PromptEnhancer\training\LoRAs\outputs\flan-t5-prompt-enhancer--2025-04-15_20-02-19\final_lora'
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")


def generate(prompt, max_new_tokens=200, num_beams=4):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    config = GenerationConfig(temperature=0.9, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=4)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=config,
        repetition_penalty=1.6,
        no_repeat_ngram_size=5
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    output = generate('Create a richly detailed scene based on this idea:\nA cat sitting in a chair', max_new_tokens=200, num_beams=4, )
    print(output)
    