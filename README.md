# LangGraph RAG QA Fine-tuning Project

This project demonstrates fine-tuning a Question Answering (QA) model using LoRA and QLoRA techniques. It includes scripts for data preprocessing, model training, and testing.

## Features

- **LoRA and QLoRA Fine-tuning:** Efficiently fine-tune `distilbert-base-uncased-distilled-squad` for QA tasks.
- **Dataset Handling:** Uses `qa_dataset.jsonl` for training and evaluation.
- **Inference Script:** A dedicated script (`src/qa_inference.py`) to load and use the fine-tuned models for answering questions.
- **Automated Fine-tuning Trigger:** A script (`src/trigger_finetune.py`) that can be integrated into a pipeline to automatically re-fine-tune the model when the dataset is updated.
- **Comprehensive Testing:** Includes unit tests for fine-tuned models and a pipeline test to verify the end-to-end flow.

## Setup and Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt # (Assuming you have a requirements.txt, if not, install transformers, datasets, peft, torch, langchain-community)
    ```

2.  **Fine-tune the Models:**
    To fine-tune the LoRA model:
    ```bash
    python3 src/fine_tune_lora_qa.py --method lora
    ```
    To fine-tune the QLoRA model:
    ```bash
    python3 src/fine_tune_qlora_qa.py --method qlora
    ```

3.  **Run Inference:**
    ```bash
    python3 src/qa_inference.py
    ```

4.  **Run Tests:**
    ```bash
    python3 test/test_models_qa.py
    python3 test/test_models_qlora_qa.py
    python3 test/test_qa_pipeline.py
    python3 test/test_lora_qa_simple_review.py
    ```

## Project Structure

```
./
├───GEMINI.md
├───.gitignore
├───README.md
├───logs/
├───models/
│   ├───lora-qa-fine-tuned-model/
│   └───qlora-qa-fine-tuned-model/
├───src/
│   ├───fine_tune_lora_qa.py
│   ├───fine_tune_qlora_qa.py
│   ├───qa_inference.py
│   ├───trigger_finetune.py
│   └───logging_config.py
├───test/
│   ├───test_models_qa.py
│   ├───test_models_qlora_qa.py
│   ├───test_qa_pipeline.py
│   └───test_lora_qa_simple_review.py
└───train_data/
    └───qa_dataset.jsonl
```

> Daily check update on 2026-01-28T14:13:57Z
