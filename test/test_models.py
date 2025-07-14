import unittest
import logging
import torch
import sys
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add the parent directory to the Python path to enable imports from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging
from langchain_community.llms import Ollama

# Setup logging
setup_logging()

class TestModels(unittest.TestCase):

    def setUp(self):
        """Set up the model, tokenizer, and reviewer LLM for testing."""
        logging.info("Setting up model, tokenizer, and reviewer LLM...")
        model_path_lora = os.path.abspath("../models/lora-fine-tuned-model")
        model_path_qlora = os.path.abspath("../models/qlora-fine-tuned-model")
        labels = ["false", "true"]
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.label2id = {label: i for i, label in enumerate(labels)}

        self.lora_model = AutoModelForSequenceClassification.from_pretrained(
            model_path_lora, 
            num_labels=2, 
            id2label=self.id2label, 
            label2id=self.label2id
        )
        self.lora_tokenizer = AutoTokenizer.from_pretrained(model_path_lora)

        self.qlora_model = AutoModelForSequenceClassification.from_pretrained(
            model_path_qlora, 
            num_labels=2, 
            id2label=self.id2label, 
            label2id=self.label2id
        )
        self.qlora_tokenizer = AutoTokenizer.from_pretrained(model_path_qlora)

        self.reviewer_llm = Ollama(model="deepseek-r1:1.5b")
        logging.info("Model, tokenizer, and reviewer LLM set up.")

    def get_prediction(self, model, tokenizer, question):
        """Tokenizes the input question, gets the model's prediction, and returns the predicted label and probabilities."""
        inputs = tokenizer(question, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0].tolist()
        predicted_class_id = logits.argmax().item()
        return model.config.id2label[predicted_class_id], probabilities

    def get_reviewer_assertion(self, prediction, expected_answer):
        """Directly asserts the prediction against the expected answer."""
        return prediction.lower() == expected_answer.lower()

    def test_lora_true_response(self):
        """Test the LoRA model with a prompt that should return 'true'."""
        logging.info("Testing LoRA model for a 'true' response...")
        question = "Is the sky blue?"
        prediction, probabilities = self.get_prediction(self.lora_model, self.lora_tokenizer, question)
        logging.info(f"LoRA model response: {prediction}, Probabilities: {probabilities}")
        self.assertTrue(self.get_reviewer_assertion(prediction, 'true'))

    def test_lora_false_response(self):
        """Test the LoRA model with a prompt that should return 'false'."""
        logging.info("Testing LoRA model for a 'false' response...")
        question = "Is the grass red?"
        prediction, probabilities = self.get_prediction(self.lora_model, self.lora_tokenizer, question)
        logging.info(f"LoRA model response: {prediction}, Probabilities: {probabilities}")
        self.assertTrue(self.get_reviewer_assertion(prediction, 'false'))

    def test_qlora_true_response(self):
        """Test the QLoRA model with a prompt that should return 'true'."""
        logging.info("Testing QLoRA model for a 'true' response...")
        question = "Is the sun hot?"
        prediction, probabilities = self.get_prediction(self.qlora_model, self.qlora_tokenizer, question)
        logging.info(f"QLoRA model response: {prediction}, Probabilities: {probabilities}")
        self.assertTrue(self.get_reviewer_assertion(prediction, 'true'))

    def test_qlora_false_response(self):
        """Test the QLoRA model with a prompt that should return 'false'."""
        logging.info("Testing QLoRA model for a 'false' response...")
        question = "Is the moon made of cheese?"
        prediction, probabilities = self.get_prediction(self.qlora_model, self.qlora_tokenizer, question)
        logging.info(f"QLoRA model response: {prediction}, Probabilities: {probabilities}")
        self.assertTrue(self.get_reviewer_assertion(prediction, 'false'))

if __name__ == '__main__':
    unittest.main()
