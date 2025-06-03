import os
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer)

# 1. Load merged JSONL dataset
dataset = load_dataset("json", data_files="all_personas_qa.jsonl")["train"]

# 2. Format for GPT-Neo: concatenate prompt and response
def format_example(example):
    return {"text": example["prompt"] + "\n" + example["response"]}

dataset = dataset.map(format_example)

# 3. Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. Tokenize the data
def tokenize_function(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. Set up training arguments
training_args = TrainingArguments(
    output_dir="./gptneo-persona-finetune",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    report_to="none"
)

# 6. Set up Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

# 7. Save model and tokenizer
trainer.save_model("./gptneo-persona-finetune")
tokenizer.save_pretrained("./gptneo-persona-finetune")