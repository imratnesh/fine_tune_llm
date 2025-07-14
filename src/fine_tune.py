
import logging
import torch
from src.logging_config import setup_logging

setup_logging()
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load the dataset
dataset = load_dataset('csv', data_files='train_data/qa_dataset.csv')

# Load the tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["answer"],
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
        start_char = answer.find(answer)
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


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)

# Define training arguments
training_args = TrainingArguments(
    output_dir="../results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"], # Using train set for evaluation for simplicity
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("../models/fine-tuned-model")
tokenizer.save_pretrained("../models/fine-tuned-model")
