import os
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# ==== CONFIGURATION ====
PERSONAS= [
    {"name": "basquiat", "data_path": "models/basquiat_data.csv", "epochs": 20},
    {"name": "cobain", "data_path": "models/cobain_data.csv"},
    {"name": "morrison", "data_path": "models/morrison_data.csv"},
    {"name": "warhol", "data_path": "models/warhol_data.csv"}
]
MODEL_NAME = "models/EleutherAI-gpt-neo-125M"
MAX_LENGTH = 128

for persona in PERSONAS:
    PERSONA = persona["name"]
    DATA_PATH = persona["data_path"]
    EPOCHS = persona.get("epochs", 10)
    MODEL_OUTPUT_DIR = f"models/{PERSONA}/"

    # ==== DATA PREPARATION ====
    df = pd.read_csv(DATA_PATH, encoding="utf-8", delimiter=";")
    df.columns = df.columns.str.strip()
    df["processed_text"] = df['Content'].apply(lambda x: f"<|{PERSONA}|> {x} <|endoftext|>")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    def create_dataset(dataframe):
        return Dataset.from_dict({"text": dataframe["processed_text"].tolist()})

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)

    # ==== TOKENIZER & MODEL ====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ==== TRAINING ====
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        save_total_limit=2,
        no_cuda=not torch.cuda.is_available(),
    )

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    print(f"Starting training for persona: {PERSONA}")
    trainer.train()

    # ==== SAVE ====
    trainer.save_model(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Model & tokenizer saved to {MODEL_OUTPUT_DIR}")