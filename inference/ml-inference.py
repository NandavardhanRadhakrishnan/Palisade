# inference.py
import joblib


def mlApproach(prompt_text):
    # Load the trained model and vectorizer
    model = joblib.load('prompt_injection_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform the input text
    prompt_tfidf = vectorizer.transform([prompt_text])

    # Predict whether it's a prompt injection
    prediction = model.predict(prompt_tfidf)

    return bool(prediction)


if __name__ == "__main__":
    # Example of using the inference function
    prompt = input("Enter a prompt: ")
    result = mlApproach(prompt)
    print(result)
