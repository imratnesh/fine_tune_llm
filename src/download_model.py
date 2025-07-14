from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

tokenizer.save_pretrained("../models/local_hf_model")
model.save_pretrained("../models/local_hf_model")

print(f"Model {model_name} downloaded and saved to ../models/local_hf_model")