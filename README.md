# Movie Review Sentiment Analysis using BiLSTM

A deep learning-based sentiment analysis system that classifies movie reviews as **Positive** or **Negative** using a **Bidirectional LSTM (BiLSTM)** neural network.

The project includes custom text preprocessing, tokenization, padding, sequence modeling, model evaluation, and a Streamlit web application for real-time predictions.

---

## Live Demo

[Streamlit App]https://movie-review-sentiment-analysis-kk2shyakkonfqqeyj9ptxf.streamlit.app/
---

## Project Overview

Movie reviews contain sequential language patterns where the meaning of a word can depend on the surrounding context.

This project uses a **Bidirectional LSTM** to learn contextual relationships in movie reviews and predict their sentiment.

### Pipeline

```text
Movie Review
     ↓
Text Cleaning
     ↓
Tokenization
     ↓
Sequence Conversion
     ↓
Padding
     ↓
Embedding
     ↓
Bidirectional LSTM
     ↓
Dropout
     ↓
Sigmoid Output
     ↓
Positive / Negative

Dataset

The project uses the Stanford IMDb Large Movie Review Dataset.

50,000 labeled movie reviews
25,000 training reviews
25,000 testing reviews
Binary sentiment classification
Positive = 1
Negative = 0

The original dataset was used so that preprocessing and tokenization could be performed as part of the project.

Data Preprocessing

The following preprocessing steps were performed:

Converted text to lowercase
Removed HTML <br> tags
Removed unnecessary characters
Preserved apostrophes in contractions such as don't and can't
Removed extra whitespace
Checked for missing reviews
Checked for duplicate reviews
Removed cross train-test duplicate reviews
Created a validation set from the training data
Train / Validation / Test
Training:   22,500
Validation:  2,500
Testing:    24,877

Tokenization

A Keras Tokenizer was used to convert text into integer sequences.

Configuration:

Vocabulary Size: 20,000
OOV Token: <OOV>
Maximum Sequence Length: 500
Padding: Post
Truncation: Post

The tokenizer was fitted only on the training data to prevent information leakage.

Model Architecture

The final model uses a Bidirectional LSTM:

Embedding
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.5)
    ↓
Dense (1, Sigmoid)
Configuration
Component	Configuration
Vocabulary	20,000
Embedding Dimension	128
LSTM Units	64
LSTM Type	Bidirectional
Dropout	0.5
Output	Sigmoid
Loss	Binary Crossentropy
Optimizer	Adam

Model Performance

Final saved model performance on the held-out test set:

Test Accuracy: 84.57%

Classification Report
Class	Precision	Recall	F1-Score
Negative	0.82	0.89	0.85
Positive	0.88	0.80	0.84
Macro Avg	0.85	0.85	0.85

The model performs reasonably well on both sentiment classes, although mixed or contextually ambiguous reviews remain more challenging.

Example Predictions
Positive Review
I absolutely loved this movie. The acting was fantastic
and the story was amazing.

Prediction:

Positive
Confidence: 91.32%
Negative Review
This movie was terrible. The story was boring
and the acting was awful.

Prediction:

Negative
Confidence: 98.94%
Streamlit Application

The project includes an interactive Streamlit interface where users can:

Enter a movie review
Analyze its sentiment
View Positive/Negative prediction
View model confidence
View model architecture information

Streamlit Application

The project includes an interactive Streamlit interface where users can:

Enter a movie review
Analyze its sentiment
View Positive/Negative prediction
View model confidence
View model architecture information


📁 Project Structure
Movie-review-Sentiment-Analysis/
│
├── artifacts/
│   ├── sentiment_bilstm.keras
│   └── tokenizer.pkl
│
├── data/
│   └── aclImdb/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL
cd Movie-review-Sentiment-Analysis

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

Command Line Prediction

You can also test the saved model directly:

python src/predict.py

Enter a movie review when prompted.

Technologies Used
Python
TensorFlow / Keras
NumPy
Scikit-learn
Streamlit
Regular Expressions
Git & GitHub
Deep Learning
Word Embeddings
RNN concepts
LSTM
Bidirectional LSTM
Dropout
Binary Classification

Key Learning Outcomes

Through this project, I implemented:

Text preprocessing from raw reviews
Tokenization and vocabulary creation
Sequence representation
Padding and truncation
Train/validation/test splitting
Embedding layers
LSTM-based sequence modeling
Bidirectional recurrent networks
Overfitting detection
Model evaluation
Model serialization
Real-time inference
Streamlit deployment

Limitations

The model performs binary classification and therefore does not explicitly predict a neutral sentiment.

Reviews containing sarcasm, mixed opinions, or subtle sentiment may be more difficult to classify accurately.