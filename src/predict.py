import os
import pickle
import tensorflow as tf

from src.preprocessing import clean_text, tokenize_and_pad


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "sentiment_bilstm.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "artifacts",
    "tokenizer.pkl"
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_sentiment(text):

    # 1. Clean text
    cleaned_text = clean_text(text)

    # 2. Tokenize + pad
    padded_sequence = tokenize_and_pad(
        tokenizer,
        [cleaned_text]
    )

    # 3. Predict
    probability = model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # 4. Convert probability to sentiment
    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    return sentiment, float(confidence)


# --------------------------------------------------
# Test prediction
# --------------------------------------------------

if __name__ == "__main__":

    review = input("\nEnter a movie review: ")

    sentiment, confidence = predict_sentiment(review)

    print("\nPrediction:", sentiment)
    print("Confidence:", f"{confidence * 100:.2f}%")