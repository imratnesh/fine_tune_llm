
import logging
import torch
from src.logging_config import setup_logging
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# Setup logging
setup_logging()

def fine_tune(method='lora'):
    """
    Fine-tunes a model using either LoRA or QLoRA for sequence classification.
    :param method: 'lora' or 'qlora'
    """
    # Load the dataset
    logging.info("Loading dataset...")
    dataset = load_dataset('csv', data_files='train_data/qa_dataset_boolean.csv')

    # Load the tokenizer and model
    model_name = "distilbert-base-uncased"
    logging.info(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define labels
    labels = ["false", "true"]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}

    if method == 'qlora':
        # QLoRA specific configuration
        logging.info("Configuring for QLoRA...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            num_labels=2,
            id2label=id2label,
            label2id=label2id
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # LoRA configuration
        logging.info("Configuring for LoRA...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2, 
            id2label=id2label, 
            label2id=label2id
        )

    # PEFT configuration for LoRA/QLoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    logging.info("PEFT model created.")

    # Preprocess the dataset
    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples["question"], truncation=True, padding=True)
        tokenized_inputs["labels"] = [label2id[str(ans).lower()] for ans in examples["answer"]]
        return tokenized_inputs

    logging.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    # Define training arguments
    output_dir = f"../logs/{method}-results"
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,
    )

    # Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
    )

    # Fine-tune the model
    logging.info(f"Starting {method} fine-tuning...")
    trainer.train()
    logging.info("Fine-tuning complete.")

    # Save the fine-tuned model
    model_path = f"../models/{method}-fine-tuned-model"
    logging.info(f"Saving model to {model_path}...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA or QLoRA.")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "qlora"], help="Fine-tuning method: lora or qlora.")
    args = parser.parse_args()
    fine_tune(args.method)
