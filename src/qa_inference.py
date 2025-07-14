import logging
import torch
import sys
import os
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Add the parent directory to the Python path to enable imports from 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging

# Setup logging
setup_logging()

def answer_question(question: str, context: str, model_type: str = "lora"):
    """
    Loads a fine-tuned QA model and answers a question.
    :param question: The question to answer.
    :param context: The context in which to find the answer.
    :param model_type: 'lora' or 'qlora' to specify which model to load.
    :return: The predicted answer.
    """
    model_path = f"../models/{model_type}-qa-fine-tuned-model"
    
    if not os.path.exists(model_path):
        logging.error(f"Model path {model_path} does not exist. Please fine-tune the model first.")
        return "Error: Model not found."

    try:
        logging.info(f"Loading {model_type} QA model from {model_path}...")
        model = AutoModelForQuestionAnswering.from_pretrained(model_path, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

        qa_input = {
            'question': question,
            'context': context
        }
        
        result = qa_pipeline(qa_input)
        predicted_answer = result['answer']
        logging.info(f"Question: {question}")
        logging.info(f"Context: {context}")
        logging.info(f"Predicted Answer: {predicted_answer}")
        return predicted_answer

    except Exception as e:
        logging.error(f"Error loading or using the model: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    # Example usage
    question = "What are your business hours?"
    context = "Our business hours are Monday to Friday, 9 AM to 5 PM EST."
    
    # You can change 'lora' to 'qlora' to test the QLoRA model
    answer = answer_question(question, context, model_type="lora")
    print(f"Answer: {answer}")

    question_new = "How do I contact customer support?"
    context_new = "You can reach customer support via email at support@example.com or by calling our toll-free number at 1-800-555-1234 during business hours."
    answer_new = answer_question(question_new, context_new, model_type="lora")
    print(f"Answer: {answer_new}")