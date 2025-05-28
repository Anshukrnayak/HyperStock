from transformers import pipeline
import pandas as pd

nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def answer_query(query, transactions):
    try:
        df = pd.DataFrame(transactions, columns=["ID", "Description", "Amount", "Category", "Date"])
        context = df[["Description", "Amount", "Category"]].to_string()
        result = nlp(question=query, context=context)
        return result["answer"]
    except Exception as e:
        print(f"Error in answer_query: {e}")
        return "Error processing query. Please try again."
