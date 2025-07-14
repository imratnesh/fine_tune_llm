from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained("../models/local_ragas_llm")
model.save_pretrained("../models/local_ragas_llm")

print(f"Model {model_name} downloaded and saved to ../models/local_ragas_llm")