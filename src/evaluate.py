import os
import logging
import json
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy
from transformers import pipeline
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Set up logging
from src.logging_config import setup_logging

setup_logging()


# 1. Load the fine-tuned model
logging.info("Loading the fine-tuned model...")
qa_pipeline = pipeline("question-answering", model="../models/local_hf_model")
logging.info("Fine-tuned model loaded.")

# 2. Create a LangGraph pipeline
class GraphState(TypedDict):
    question: str
    answer: str

def answer_question(state):
    question = state["question"]
    logging.info(f"Answering question: {question}")
    result = qa_pipeline(question=question, context=question)
    answer = result['answer']
    logging.info(f"Generated answer: {answer}")
    return {"answer": answer}

workflow = StateGraph(GraphState)
workflow.add_node("answer_question", answer_question)
workflow.set_entry_point("answer_question")
workflow.add_edge("answer_question", END)
app = workflow.compile()

# 3. Load the dataset for evaluation
logging.info("Loading the dataset for evaluation...")
import pandas as pd
df = pd.read_csv('train_data/qa_dataset.csv')
questions = df["question"].tolist()
ground_truths = df["answer"].tolist()
logging.info("Dataset loaded.")

# 4. Get answers from the fine-tuned model
logging.info("Getting answers from the fine-tuned model...")
answers = []
for question in questions:
    # The qa_pipeline should generate answers based on the question
    result = qa_pipeline(question=question, context=question)
    answers.append(result['answer'])
logging.info("Finished getting answers.")

# 5. Evaluate with Ragas
logging.info("Evaluating with Ragas...")
data = {
    "question": questions,
    "answer": answers,
    "ground_truth": ground_truths,
    "retrieved_contexts": [[q] for q in questions]
}
dataset = Dataset.from_dict(data)

# Use Ollama for evaluation
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("../models/local_ragas_llm")
model = AutoModelForSeq2SeqLM.from_pretrained("../models/local_ragas_llm")
ragas_llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
ragas_llm = HuggingFacePipeline(pipeline=ragas_llm_pipeline)

# For embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

result = evaluate(
    dataset = dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=embedding_model,
    run_config=RunConfig(timeout=600)
)

logging.info(f"Ragas evaluation result: {result}")

# Save result to a text file
with open("../logs/evaluation_results.txt", "w") as f:
    f.write(str(result))

logging.info("Evaluation results saved to evaluation_results.txt")
print(result)
