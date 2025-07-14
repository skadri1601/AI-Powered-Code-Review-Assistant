# scripts/finetune.py

"""
Fine-tune a CodeT5 model on diff→review pairs using HF Trainer.
"""

import os
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# ─── Config ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "hf_dataset"

MODEL_CHECKPOINT = "Salesforce/codet5-base"
OUTPUT_DIR       = PROJECT_ROOT / "checkpoints" / "codet5-finetuned"
BATCH_SIZE       = 4
NUM_EPOCHS       = 3
MAX_SOURCE_LEN   = 256
MAX_TARGET_LEN   = 128

# ─── Load & Prepare Dataset ────────────────────────────────────────────

def load_and_tokenize():
    # 1) Load JSONL files into a DatasetDict
    data_files = {
        "train":      str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "validation.jsonl"),
    }
    raw_datasets = load_dataset("json", data_files=data_files)

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

    # 3) Tokenization function
    def preprocess_fn(batch):
        inputs  = batch["input"]
        targets = batch["target"]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 4) Map & format
    tokenized = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=["input", "target"]
    )

    return tokenized, tokenizer, model

# ─── Train ─────────────────────────────────────────────────────────────

def main():
    tokenized_datasets, tokenizer, model = load_and_tokenize()

    # 5) Data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model
    )

    # 6) Training arguments (compatible with older transformer versions)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        save_total_limit=2,
        predict_with_generate=True,
        logging_steps=50,
        save_steps=200,
        eval_steps=200
    )

    # 7) Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 8) Launch training
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))


if __name__ == "__main__":
    main()
