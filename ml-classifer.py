# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
data = pd.read_csv('english.csv')

# Split data into text and labels
X = data['text']
y = data['label']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Text preprocessing: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Build a classifier (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the model and vectorizer for inference
joblib.dump(model, 'prompt_injection_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
