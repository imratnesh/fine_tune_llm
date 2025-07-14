import unittest
import logging
import torch
import sys
import os
import json
import random
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from langchain_community.llms import Ollama

# Add the parent directory to the Python path to enable imports from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging

# Setup logging
setup_logging()

class TestLoraQAReview(unittest.TestCase):

    def setUp(self):
        """Set up the LoRA model, tokenizer, and reviewer LLM for testing."""
        logging.info("Setting up LoRA QA model, tokenizer, and reviewer LLM...")
        model_path_lora = "../models/lora-qa-fine-tuned-model"
        
        self.lora_qa_model = AutoModelForQuestionAnswering.from_pretrained(model_path_lora, low_cpu_mem_usage=True)
        self.lora_qa_tokenizer = AutoTokenizer.from_pretrained(model_path_lora)
        self.lora_qa_pipeline = pipeline("question-answering", model=self.lora_qa_model, tokenizer=self.lora_qa_tokenizer)
        
        self.reviewer_llm = Ollama(model="deepseek-r1:1.5b")

        # Load the dataset for ground truth
        with open(os.path.join(os.path.dirname(__file__), '..', 'train_data', 'qa_dataset.jsonl'), 'r') as f:
            self.qa_data = [json.loads(line) for line in f]

        logging.info("LoRA QA Model, tokenizer, and reviewer LLM set up.")

    def get_reviewer_assertion(self, question, predicted_answer, ground_truth_answer):
        """
        Queries the reviewer LLM to get an assertion on the predicted answer.
        The reviewer checks if the predicted answer is satisfied by the ground truth.
        """
        prompt = (
            f"Act as a reviewer. Here is a question: '{question}'. "
            f"The predicted answer is: '{predicted_answer}'. "
            f"The ground truth answer is: '{ground_truth_answer}'. "
            f"If the predicted answer is a correct and relevant span extracted from the ground truth, return True. Else return False."
            f"Respond with only 'True' or 'False'."
        )
        response = self.reviewer_llm.invoke(prompt)
        logging.info(f"Reviewer LLM response: {response.strip()}")
        return response.strip().lower() == "true"

    def test_lora_qa_response_with_ollama_review(self):
        """Test the LoRA QA model with a random question from the dataset using Ollama for review."""
        logging.info("Testing LoRA QA model response with Ollama review...")
        
        # Select a random item from the dataset
        item = random.choice(self.qa_data)
        question = item["question"]
        ground_truth_answer = item["answer"]

        qa_input = {
            'question': question,
            'context': ground_truth_answer # Using ground truth as context for consistency
        }
        
        result = self.lora_qa_pipeline(qa_input)
        predicted_answer = result['answer']

        logging.info(f"Question: {question}")
        logging.info(f"Predicted Answer: {predicted_answer}")
        logging.info(f"Ground Truth: {ground_truth_answer}")

        self.assertTrue(
            self.get_reviewer_assertion(question, predicted_answer, ground_truth_answer),
            f"Reviewer assertion failed for question: {question}"
        )

if __name__ == '__main__':
    unittest.main()