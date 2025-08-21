import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- 1. Load the Saved Model and Tokenizer ---

# FIX: Updated the path to point to the correct high-accuracy model folder.
model_dir = "./sentiment_model_distilbert_high_accuracy"

print(f"Loading model from {model_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
print("Model loaded successfully!")

# We need the same preprocessing function used during training
def preprocess_text(text):
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(lemmatized_tokens)

# --- 2. Create the Prediction Function ---

def predict_sentiment(text):
    """Takes a raw text review and predicts its sentiment."""
    # Move model to evaluation mode
    model.eval()

    # Preprocess the input text
    cleaned_text = preprocess_text(text)

    # Tokenize the text
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True)

    # Make prediction (no need to calculate gradients)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (0 for negative, 1 for positive)
    prediction = torch.argmax(outputs.logits, dim=-1).item()

    # Return the sentiment as a string
    return "Positive" if prediction == 1 else "Negative"

# --- 3. Get User Input and Predict ---

if __name__ == "__main__":
    # Example reviews to test
    review1 = "This was the best movie I have ever seen. The acting was incredible and the story was perfect!"
    review2 = "A complete waste of time. The plot was boring and the characters were unlikeable."

    print("\n--- Testing Examples ---")
    print(f"Review: '{review1}'\nSentiment: {predict_sentiment(review1)}\n")
    print(f"Review: '{review2}'\nSentiment: {predict_sentiment(review2)}\n")

    # Interactive loop for custom input
    print("--- Enter Your Own Review ---")
    while True:
        user_input = input("Type a movie review (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}\n")