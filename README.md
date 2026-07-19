# Sentiment Analysis

A small text-classification project that predicts whether a movie review is positive
or negative, built to practice a classic NLP pipeline end to end.

## What it does

- Loads the NLTK `movie_reviews` corpus (2,000 labeled reviews).
- Converts review text into word-count features with `CountVectorizer`
  (top 2,000 words).
- Trains a Multinomial Naive Bayes classifier on an 80/20 train/test split.
- Reports accuracy and a full classification report (precision/recall/F1).
- Includes a `predict_sentiment()` helper to classify new, arbitrary text.

## Tech stack

- Python
- NLTK
- pandas
- scikit-learn (`CountVectorizer`, `MultinomialNB`)

## Running it

Open `Sentiment Analysis.ipynb` in Jupyter and run the cells in order. The first
run downloads the `movie_reviews` corpus via `nltk.download`.
