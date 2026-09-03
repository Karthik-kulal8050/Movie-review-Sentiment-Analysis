import os
import pickle

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer

from src.preprocessing import clean_text, tokenize_and_pad,SEED


# ============================================================
# 1. Configuration
# ============================================================

DATA_DIR = "E:/Movie-review-Sentiment-Analysis/data/aclImdb"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

ARTIFACTS_DIR = "E:/Movie-review-Sentiment-Analysis/artifacts"

VOCAB_SIZE = 20000
EMBEDDING_DIM = 128
LSTM_UNITS = 64
MAX_LEN = 500
BATCH_SIZE = 64
EPOCHS = 10
RANDOM_STATE = SEED


# ============================================================
# 2. Load reviews
# ============================================================

def load_reviews(folder_path, label, split):
    reviews = []

    for filename in os.listdir(folder_path):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            review = f.read()

        reviews.append({
            "review": review,
            "sentiment": label,
            "split": split
        })

    return reviews


print("Loading dataset...")

positive_train = load_reviews(
    os.path.join(TRAIN_DIR, "pos"),
    1,
    "train"
)

negative_train = load_reviews(
    os.path.join(TRAIN_DIR, "neg"),
    0,
    "train"
)

positive_test = load_reviews(
    os.path.join(TEST_DIR, "pos"),
    1,
    "test"
)

negative_test = load_reviews(
    os.path.join(TEST_DIR, "neg"),
    0,
    "test"
)


data = (
    positive_train
    + negative_train
    + positive_test
    + negative_test
)

df = pd.DataFrame(data)

print(f"Total reviews loaded: {len(df)}")


# ============================================================
# 3. Remove train-test duplicate reviews
# ============================================================

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

train_reviews = set(train_df["review"])
test_reviews = set(test_df["review"])

cross_split_duplicates = train_reviews.intersection(test_reviews)

print(
    f"Cross train-test duplicates found: "
    f"{len(cross_split_duplicates)}"
)

test_df = test_df[
    ~test_df["review"].isin(cross_split_duplicates)
].copy()

print(f"Training reviews: {len(train_df)}")
print(f"Testing reviews after cleanup: {len(test_df)}")


# ============================================================
# 4. Clean text
# ============================================================

print("\nCleaning text...")

train_df["cleaned_review"] = train_df["review"].apply(clean_text)
test_df["cleaned_review"] = test_df["review"].apply(clean_text)


# ============================================================
# 5. Train / Validation split
# ============================================================

X = train_df["cleaned_review"]
y = train_df["sentiment"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y
)

X_test = test_df["cleaned_review"]
y_test = test_df["sentiment"]

print("\nDataset split:")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")


# ============================================================
# 6. Tokenization
# ============================================================

print("\nFitting tokenizer...")

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

# IMPORTANT:
# Fit tokenizer ONLY on training data
tokenizer.fit_on_texts(X_train)

print(f"Total vocabulary learned: {len(tokenizer.word_index)}")


# ============================================================
# 7. Tokenization + Padding
# ============================================================

print("\nConverting text to sequences...")

X_train_pad = tokenize_and_pad(
    tokenizer,
    X_train,
    MAX_LEN
)

X_val_pad = tokenize_and_pad(
    tokenizer,
    X_val,
    MAX_LEN
)

X_test_pad = tokenize_and_pad(
    tokenizer,
    X_test,
    MAX_LEN
)

print("Padded shapes:")
print("Train      :", X_train_pad.shape)
print("Validation :", X_val_pad.shape)
print("Test       :", X_test_pad.shape)


# ============================================================
# 8. Build BiLSTM model
# ============================================================

print("\nBuilding model...")

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS)
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


# ============================================================
# 9. Compile
# ============================================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# 10. Early stopping
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# ============================================================
# 11. Train
# ============================================================

print("\nTraining model...")

history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)


# ============================================================
# 12. Final evaluation
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    X_test_pad,
    y_test,
    batch_size=BATCH_SIZE
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# ============================================================
# 13. Classification report
# ============================================================

y_prob = model.predict(
    X_test_pad,
    batch_size=BATCH_SIZE
)

y_pred = (y_prob >= 0.5).astype(int).flatten()

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"]
    )
)


# ============================================================
# 14. Confusion matrix
# ============================================================

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. Save model
# ============================================================

os.makedirs(
    ARTIFACTS_DIR,
    exist_ok=True
)

model_path = os.path.join(
    ARTIFACTS_DIR,
    "sentiment_bilstm.keras"
)

model.save(model_path)

print(f"\nModel saved to: {model_path}")


# ============================================================
# 16. Save tokenizer
# ============================================================

tokenizer_path = os.path.join(
    ARTIFACTS_DIR,
    "tokenizer.pkl"
)

with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)

print(f"Tokenizer saved to: {tokenizer_path}")

print("\nTraining pipeline completed successfully!")