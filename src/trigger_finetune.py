import os
import logging
import subprocess
import json
from datetime import datetime

# Add the parent directory to the Python path to enable imports from 'src'
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import setup_logging

# Setup logging
setup_logging()

def get_last_modified_time(filepath):
    """
    Returns the last modification time of a file.
    """
    if os.path.exists(filepath):
        return os.path.getmtime(filepath)
    return 0

def trigger_finetune(model_type: str = "lora"):
    """
    Triggers fine-tuning if the dataset is newer than the last fine-tuned model.
    :param model_type: 'lora' or 'qlora'
    """
    dataset_path = "train_data/qa_dataset.jsonl"
    model_path = f"models/{model_type}-qa-fine-tuned-model"
    finetune_script = f"src/fine_tune_{model_type}_qa.py"

    dataset_modified_time = get_last_modified_time(dataset_path)
    model_modified_time = get_last_modified_time(model_path)

    logging.info(f"Dataset last modified: {datetime.fromtimestamp(dataset_modified_time)}")
    logging.info(f"Model last modified: {datetime.fromtimestamp(model_modified_time)}")

    if dataset_modified_time > model_modified_time:
        logging.info(f"Dataset is newer than {model_type} model. Triggering fine-tuning...")
        try:
            command = ["python3", finetune_script, "--method", model_type]
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info(f"Fine-tuning output:\n{process.stdout}")
            if process.stderr:
                logging.error(f"Fine-tuning errors:\n{process.stderr}")
            logging.info(f"{model_type} fine-tuning completed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Error during {model_type} fine-tuning: {e}\nStdout: {e.stdout}\nStderr: {e.stderr}")
            return False
    else:
        logging.info(f"Dataset is not newer than {model_type} model. No fine-tuning needed.")
        return False

if __name__ == "__main__":
    # Example usage
    trigger_finetune(model_type="lora")
    trigger_finetune(model_type="qlora")