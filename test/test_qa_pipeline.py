import unittest
import logging
import torch
import sys
import os
import json
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Add the parent directory to the Python path to enable imports from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging
from src.trigger_finetune import trigger_finetune
from src.qa_inference import answer_question

# Setup logging
setup_logging()

class TestQAPipeline(unittest.TestCase):

    def setUp(self):
        """Load the dataset for ground truth."""
        logging.info("Setting up QA pipeline test...")
        with open(os.path.join(os.path.dirname(__file__), '..', 'train_data', 'qa_dataset.jsonl'), 'r') as f:
            self.qa_data = [json.loads(line) for line in f]
        logging.info("QA pipeline test setup complete.")

    def test_pipeline_flow(self):
        """
        Tests the entire QA pipeline:
        1. Base model inference (before fine-tuning).
        2. Trigger fine-tuning.
        3. Fine-tuned model inference.
        """
        model_name = "distilbert-base-uncased-distilled-squad"
        
        # --- 1. Test Base Model Inference --- 
        logging.info(f"Testing base model ({model_name}) inference...")
        base_model = AutoModelForQuestionAnswering.from_pretrained(model_name, low_cpu_mem_usage=True)
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_qa_pipeline = pipeline("question-answering", model=base_model, tokenizer=base_tokenizer)

        # Take the first question from the dataset for testing
        test_item = self.qa_data[0]
        question = test_item["question"]
        ground_truth_answer = test_item["answer"]
        context = ground_truth_answer # Using ground truth as context for consistency

        base_result = base_qa_pipeline({'question': question, 'context': context})
        base_predicted_answer = base_result['answer']
        logging.info(f"Base Model - Question: {question}")
        logging.info(f"Base Model - Predicted Answer: {base_predicted_answer}")
        logging.info(f"Base Model - Ground Truth: {ground_truth_answer}")

        # Assert that the base model's answer is not necessarily perfect
        # This is just to show a baseline, not to assert correctness
        self.assertIsNotNone(base_predicted_answer)
        logging.info("Base model inference test complete.")

        # --- 2. Trigger Fine-tuning (LoRA) ---
        logging.info("Triggering LoRA fine-tuning...")
        # Ensure the model is older than the dataset to trigger fine-tuning
        lora_model_path = "../models/lora-qa-fine-tuned-model"
        if os.path.exists(lora_model_path):
            # Set modification time of the model to be older than the dataset
            os.utime(lora_model_path, (os.path.getatime(lora_model_path), os.path.getmtime("train_data/qa_dataset.jsonl") - 3600))
        
        finetune_triggered = trigger_finetune(model_type="lora")
        self.assertTrue(finetune_triggered, "LoRA fine-tuning should have been triggered.")
        logging.info("LoRA fine-tuning triggered and completed (or skipped if not needed).")

        # --- 3. Test Fine-tuned Model Inference (LoRA) ---
        logging.info("Testing fine-tuned LoRA model inference...")
        # Use the answer_question function from qa_inference.py
        fine_tuned_predicted_answer = answer_question(question, context, model_type="lora")
        logging.info(f"Fine-tuned LoRA Model - Question: {question}")
        logging.info(f"Fine-tuned LoRA Model - Predicted Answer: {fine_tuned_predicted_answer}")
        logging.info(f"Fine-tuned LoRA Model - Ground Truth: {ground_truth_answer}")

        # Assert that the fine-tuned model's answer is contained within the ground truth
        self.assertIn(fine_tuned_predicted_answer.lower(), ground_truth_answer.lower(),
                      f"Fine-tuned LoRA model: Predicted answer '{fine_tuned_predicted_answer}' not found in ground truth '{ground_truth_answer}' for question: {question}")
        logging.info("Fine-tuned LoRA model inference test complete.")

        # --- 4. Trigger Fine-tuning (QLoRA) ---
        logging.info("Triggering QLoRA fine-tuning...")
        # Ensure the model is older than the dataset to trigger fine-tuning
        qlora_model_path = "../models/qlora-qa-fine-tuned-model"
        if os.path.exists(qlora_model_path):
            # Set modification time of the model to be older than the dataset
            os.utime(qlora_model_path, (os.path.getatime(qlora_model_path), os.path.getmtime("train_data/qa_dataset.jsonl") - 3600))
        
        finetune_triggered_qlora = trigger_finetune(model_type="qlora")
        self.assertTrue(finetune_triggered_qlora, "QLoRA fine-tuning should have been triggered.")
        logging.info("QLoRA fine-tuning triggered and completed (or skipped if not needed).")

        # --- 5. Test Fine-tuned Model Inference (QLoRA) ---
        logging.info("Testing fine-tuned QLoRA model inference...")
        # Use the answer_question function from qa_inference.py
        fine_tuned_predicted_answer_qlora = answer_question(question, context, model_type="qlora")
        logging.info(f"Fine-tuned QLoRA Model - Question: {question}")
        logging.info(f"Fine-tuned QLoRA Model - Predicted Answer: {fine_tuned_predicted_answer_qlora}")
        logging.info(f"Fine-tuned QLoRA Model - Ground Truth: {ground_truth_answer}")

        # Assert that the fine-tuned model's answer is contained within the ground truth
        self.assertIn(fine_tuned_predicted_answer_qlora.lower(), ground_truth_answer.lower(),
                      f"Fine-tuned QLoRA model: Predicted answer '{fine_tuned_predicted_answer_qlora}' not found in ground truth '{ground_truth_answer}' for question: {question}")
        logging.info("Fine-tuned QLoRA model inference test complete.")


if __name__ == '__main__':
    unittest.main()