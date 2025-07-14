

import pandas as pd
import json
import os

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    df = pd.read_csv(csv_file_path)
    with open(jsonl_file_path, 'w') as f:
        for index, row in df.iterrows():
            # Assuming your CSV has 'question' and 'answer' columns
            data = {"question": row["question"], "answer": row["answer"]}
            f.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    # Define paths relative to the project root
    csv_input_path = 'train_data/qa_dataset.csv'
    jsonl_output_path = 'train_data/qa_dataset.jsonl'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(jsonl_output_path), exist_ok=True)

    convert_csv_to_jsonl(csv_input_path, jsonl_output_path)
    print(f"Successfully converted {csv_input_path} to {jsonl_output_path}")

