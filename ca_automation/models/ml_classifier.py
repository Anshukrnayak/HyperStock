# models/ml_classifier.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

# Training data
data = pd.DataFrame({
    'description': [
        'office supplies', 'client payment', 'rent', 'salary',
        'sales revenue', 'utility bill', 'consulting fees', 'travel expenses',
        'software subscription', 'dividend income', 'marketing', 'equipment'
    ],
    'category': [
        'expense', 'income', 'expense', 'expense',
        'income', 'expense', 'income', 'expense',
        'expense', 'income', 'expense', 'expense'
    ]
})

# Initialize and train model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['description'])
y = data['category']
model = MultinomialNB()
model.fit(X, y)

def predict_category(description):
    try:
        X_new = vectorizer.transform([description])
        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max()
        return prediction, confidence
    except Exception as e:
        print(f"Error in predict_category: {e}")
        return "expense", 0.5

def active_learning_query(description, predicted_category):
    print(f"Active Learning: Uncertain prediction for '{description}' (Category: {predicted_category})")
    global data, X, y, model
    new_data = pd.DataFrame({'description': [description], 'category': [predicted_category]})
    data = pd.concat([data, new_data], ignore_index=True)
    X = vectorizer.fit_transform(data['description'])
    y = data['category']
    model.fit(X, y)
