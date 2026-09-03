import re
import tensorflow as tf
import numpy as np
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Image-like constant isn't needed here; this is text preprocessing.
MAX_LEN = 500


def clean_text(text):
    """
    Clean a movie review before tokenization.

    Steps:
    1. Convert text to lowercase
    2. Remove HTML tags
    3. Keep letters, apostrophes and spaces
    4. Remove extra whitespace

    Example:
        "I DON'T like this movie!<br />"
        -> "i don't like this movie"
    """

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags such as <br />
    text = re.sub(r"<br\s*/?>", " ", text)

    # Keep letters, apostrophes and spaces
    text = re.sub(r"[^a-z'\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_and_pad(tokenizer, texts, max_len=MAX_LEN):
    """
    Convert text into integer sequences and pad/truncate them.

    Parameters:
        tokenizer: Fitted Keras tokenizer
        texts: Text reviews
        max_len: Maximum sequence length

    Returns:
        Padded NumPy array
    """

    sequences = tokenizer.texts_to_sequences(texts)

    padded_sequences = tf.keras.utils.pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    return padded_sequences