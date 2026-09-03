import streamlit as st

from src.predict import predict_sentiment


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and our BiLSTM model will "
    "predict whether the sentiment is positive or negative."
)


# --------------------------------------------------
# Model Information
# --------------------------------------------------

with st.expander("Model Information"):

    st.write("**Model:** Bidirectional LSTM (BiLSTM)")

    st.write("**Embedding Dimension:** 128")

    st.write("**LSTM Units:** 64")

    st.write("**Maximum Sequence Length:** 500")

    st.write("**Vocabulary Size:** 20,000")

    st.write("**Task:** Binary Sentiment Classification")

    st.write("**Dataset:** Stanford IMDb Large Movie Review Dataset")


# --------------------------------------------------
# Review Input
# --------------------------------------------------

review = st.text_area(
    "Enter your movie review",
    placeholder=(
        "Example: This movie was absolutely fantastic. "
        "The acting and story were amazing!"
    ),
    height=180
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Analyze Sentiment", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        try:

            sentiment, confidence = predict_sentiment(review)

            st.divider()

            # ------------------------------------------
            # Result
            # ------------------------------------------

            if sentiment == "Positive":

                st.success(
                    f"### Positive Review\n\n"
                    f"**Confidence: {confidence * 100:.2f}%**"
                )

            else:

                st.error(
                    f"### Negative Review\n\n"
                    f"**Confidence: {confidence * 100:.2f}%**"
                )

            # ------------------------------------------
            # Confidence Progress Bar
            # ------------------------------------------

            st.write("### Prediction Confidence")

            st.progress(confidence)

            st.write(
                f"The model is **{confidence * 100:.2f}% confident** "
                f"in its prediction."
            )

        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Built using TensorFlow, BiLSTM, and Streamlit."
)