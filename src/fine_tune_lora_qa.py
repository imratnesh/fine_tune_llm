
import logging
import torch
import sys
import os

# Add the parent directory to the Python path to enable imports from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

# Setup logging
setup_logging()

def fine_tune_qa(method='lora'):
    """
    Fine-tunes a model using either LoRA or QLoRA for question answering.
    :param method: 'lora' or 'qlora'
    """
    # Load the dataset
    logging.info("Loading dataset...")
    dataset = load_dataset('json', data_files='train_data/qa_dataset.jsonl')

    # Load the tokenizer and model
    model_name = "distilbert-base-uncased-distilled-squad"
    logging.info(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if method == 'qlora':
        # QLoRA specific configuration
        logging.info("Configuring for QLoRA...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        # LoRA configuration
        logging.info("Configuring for LoRA...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # PEFT configuration for LoRA/QLoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="QUESTION_ANS",
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)
    logging.info("PEFT model created.")

    # Preprocess the dataset
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = [ans.strip() for ans in examples["answer"]] # Use answer as context for simplicity
        
        inputs = tokenizer(
            questions,
            contexts,
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answer"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = contexts[i].find(answer)
            end_char = start_char + len(answer)
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] < start_char:
                    idx += 1
                start_positions.append(idx)

                idx = context_end
                while idx >= context_start and offset[idx][1] > end_char:
                    idx -= 1
                end_positions.append(idx)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    logging.info("Preprocessing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

    # Define training arguments
    output_dir = f"../logs/{method}-qa-results"
    # Optimized for lower RAM usage
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=1,
    )

    # Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
    )

    # Fine-tune the model
    logging.info(f"Starting {method} QA fine-tuning...")
    trainer.train()
    logging.info("QA Fine-tuning complete.")

    # Save the fine-tuned model
    model_path = f"../models/{method}-qa-fine-tuned-model"
    logging.info(f"Saving QA model to {model_path}...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    logging.info("QA Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using LoRA or QLoRA for QA.")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "qlora"], help="Fine-tuning method: lora or qlora.")
    args = parser.parse_args()
    fine_tune_qa(args.method)
