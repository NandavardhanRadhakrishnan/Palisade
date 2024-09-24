# inference.py
import joblib


def detect_injection(prompt_text):
    # Load the trained model and vectorizer
    model = joblib.load('prompt_injection_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform the input text
    prompt_tfidf = vectorizer.transform([prompt_text])

    # Predict whether it's a prompt injection
    prediction = model.predict(prompt_tfidf)

    if prediction == 1:
        return "Potential prompt injection detected!"
    else:
        return "No prompt injection detected."


if __name__ == "__main__":
    # Example of using the inference function
    prompt = input("Enter a prompt: ")
    result = detect_injection(prompt)
    print(result)
