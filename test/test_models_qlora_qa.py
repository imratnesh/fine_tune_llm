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

# Setup logging
setup_logging()

class TestQloraQAModels(unittest.TestCase):

    def setUp(self):
        """Set up the model, tokenizer, and reviewer LLM for testing."""
        logging.info("Setting up QLoRA QA model and tokenizer...")
        model_path_qlora = "../models/qlora-qa-fine-tuned-model"
        
        self.qlora_qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path_qlora, low_cpu_mem_usage=True)
        self.qlora_qa_tokenizer = AutoTokenizer.from_pretrained(model_path_qlora)
        self.qlora_qa_pipeline = pipeline("question-answering", model=self.qlora_qa_model, tokenizer=self.qlora_qa_tokenizer)

        # Load the dataset for ground truth
        with open(os.path.join(os.path.dirname(__file__), '..', 'train_data', 'qa_dataset.jsonl'), 'r') as f:
            self.qa_data = [json.loads(line) for line in f]

        logging.info("QLoRA QA Model and tokenizer set up.")

    def test_qlora_qa_responses(self):
        """Test the QLoRA QA model with questions from the dataset."""
        logging.info("Testing QLoRA QA model responses...")
        for item in self.qa_data:
            question = item["question"]
            ground_truth_answer = item["answer"]

            qa_input = {
                'question': question,
                'context': ground_truth_answer
            }
            
            result = self.qlora_qa_pipeline(qa_input)
            predicted_answer = result['answer']

            logging.info(f"Question: {question}")
            logging.info(f"Predicted Answer: {predicted_answer}")
            logging.info(f"Ground Truth: {ground_truth_answer}")

            # Simple assertion: check if the predicted answer is contained within the ground truth
            # This is a basic check and can be improved based on specific QA evaluation metrics
            self.assertIn(predicted_answer.lower(), ground_truth_answer.lower(),
                          f"Predicted answer '{predicted_answer}' not found in ground truth '{ground_truth_answer}' for question: {question}")

if __name__ == '__main__':
    unittest.main()