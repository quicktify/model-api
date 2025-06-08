# Load slang dictionary
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import re
import string
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Literal

import absl.logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

absl.logging.set_verbosity(absl.logging.ERROR)

# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# factory = StemmerFactory()
# stemmer = factory.create_stemmer()

try:
    with open("app/dict/merged_slang_dict.json", "r", encoding="utf-8") as f:
        slang_dict = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load slang dictionary: {e}")

stop_words = {
    "yang",
    "untuk",
    "dan",
    "di",
    "ke",
    "dari",
    "ini",
    "itu",
    "dengan",
    "atau",
    "tapi",
}


class IndoTextPreprocessor:
    def __init__(self):
        self.lowercase = True
        self.remove_non_ascii = True
        self.remove_punctuation = True
        self.remove_numbers = True
        self.remove_stopwords = True
        self.remove_extra_spaces = True
        # self.stemming = False

    def normalize_slang(self, text):
        tokens = text.split()
        return " ".join(slang_dict.get(word, word) for word in tokens)

    def clean_text(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        if self.lowercase:
            text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        if self.remove_non_ascii:
            text = (
                unicodedata.normalize("NFKD", text)
                .encode("ascii", "ignore")
                .decode("utf-8", "ignore")
            )
        if self.remove_punctuation:
            for punct in string.punctuation:
                text = text.replace(punct, " ")
            text = re.sub(r"\s+", " ", text).strip()
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)
        text = self.normalize_slang(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        # if self.stemming:
        #     text = " ".join(tokens)
        #     text = stemmer.stem(text)
        #     tokens = text.split()
        cleaned_text = " ".join(tokens)
        if self.remove_extra_spaces:
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        if len(cleaned_text) <= 5 or re.fullmatch(r"(.)\1{2,}", cleaned_text):
            return ""
        return cleaned_text

    def transform(self, text_series):
        return text_series.apply(self.clean_text)


# Load Models
try:
    tokenizer = pickle.load(
        open(
            "app/models/sentiment_emotion_classification_artifacts/tokenizer.pkl", "rb"
        )
    )
    le_emosi = pickle.load(
        open("app/models/sentiment_emotion_classification_artifacts/le_emosi.pkl", "rb")
    )
    le_sentimen = pickle.load(
        open(
            "app/models/sentiment_emotion_classification_artifacts/le_sentimen.pkl",
            "rb",
        )
    )
    nn_model = tf.keras.models.load_model(
        "app/models/sentiment_emotion_classification_artifacts/nn_multitask_model.h5"
    )
    svm_multi = joblib.load(
        "app/models/sentiment_emotion_classification_artifacts/hybrid_nn_svm_model_tuned.pkl"
    )
    nb_multi = joblib.load(
        "app/models/sentiment_emotion_classification_artifacts/hybrid_nn_nb_model_tuned.pkl"
    )
    svm_basic = joblib.load(
        "app/models/sentiment_emotion_classification_artifacts/standalone_svm_model_tuned.pkl"
    )
    nb_basic = joblib.load(
        "app/models/sentiment_emotion_classification_artifacts/standalone_nb_model_tuned.pkl"
    )
except FileNotFoundError as e:
    raise RuntimeError(f"Model or tokenizer file missing: {e}")

# Extract individual models
svm_emosi, svm_sentimen = svm_multi.estimators_
nb_emosi, nb_sentimen = nb_multi.estimators_

# Feature Extractor
feature_extractor = tf.keras.Model(
    inputs=nn_model.input, outputs=nn_model.layers[-3].output
)


def generate_wordcloud_words(texts, top_n=50):
    if not texts:
        return []
    text = " ".join(texts)
    words = text.split()
    words = [w.lower() for w in words]  # pastikan semua lowercase
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return [{"word": w, "count": c} for w, c in most_common]


def classify_sentiment_emotion(
    reviews: List[str],
    model_type: Literal[
        "nn-only", "hybrid-svm", "hybrid-nb", "standalone-svm", "standalone-nb"
    ] = "nn-only",
    background_tasks=None,
) -> Dict[str, Any]:
    import logging

    start_time = time.time()
    # Daftar label tetap
    SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
    EMOTION_LABELS = ["Anger", "Fear", "Happy", "Love", "Neutral", "Sad"]

    df = pd.Series(reviews)
    preprocessor = IndoTextPreprocessor()
    cleaned = preprocessor.transform(df).fillna("")
    cleaned = cleaned.apply(lambda x: x if x.strip() != "" else "kosong")

    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=120, padding="post", truncating="post")

    # Feature extraction for hybrids
    features = feature_extractor.predict(padded, batch_size=32)
    features_nb = np.maximum(0, features)

    # NN Only Model
    nn_pred_emosi_prob, nn_pred_sentimen_prob = nn_model.predict(padded, batch_size=32)
    nn_pred_emosi = np.argmax(nn_pred_emosi_prob, axis=1)
    nn_pred_sentimen = np.argmax(nn_pred_sentimen_prob, axis=1)

    # Hybrid Models
    emosi_preds_svm = svm_emosi.predict(features)
    sentimen_preds_svm = svm_sentimen.predict(features)
    emosi_preds_nb = nb_emosi.predict(features_nb)
    sentimen_preds_nb = nb_sentimen.predict(features_nb)

    # Label decoding
    if model_type == "nn-only":
        emosi_labels = le_emosi.inverse_transform(nn_pred_emosi)
        sentimen_labels = le_sentimen.inverse_transform(nn_pred_sentimen)
        emosi_scores = nn_pred_emosi_prob.max(axis=1)
        sentimen_scores = nn_pred_sentimen_prob.max(axis=1)
    elif model_type == "hybrid-svm":
        emosi_labels = le_emosi.inverse_transform(emosi_preds_svm)
        sentimen_labels = le_sentimen.inverse_transform(sentimen_preds_svm)
        emosi_scores = np.ones_like(emosi_labels, dtype=float)  # Placeholder
        sentimen_scores = np.ones_like(sentimen_labels, dtype=float)  # Placeholder
    elif model_type == "hybrid-nb":
        emosi_labels = le_emosi.inverse_transform(emosi_preds_nb)
        sentimen_labels = le_sentimen.inverse_transform(sentimen_preds_nb)
        emosi_scores = np.ones_like(emosi_labels, dtype=float)  # Placeholder
        sentimen_scores = np.ones_like(sentimen_labels, dtype=float)  # Placeholder
    else:
        raise ValueError("Invalid model_type")

    # Build response with all labels always present
    sentiment_result = {label: [] for label in SENTIMENT_LABELS}
    emotion_result = {label: [] for label in EMOTION_LABELS}
    sentiment_wordclouds = {label: [] for label in SENTIMENT_LABELS}
    emotion_wordclouds = {label: [] for label in EMOTION_LABELS}
    sentiment_percentages = {label: 0 for label in SENTIMENT_LABELS}
    emotion_percentages = {label: 0 for label in EMOTION_LABELS}
    sentiment_counts = {label: 0 for label in SENTIMENT_LABELS}
    emotion_counts = {label: 0 for label in EMOTION_LABELS}

    # Isi data jika ada
    for label in SENTIMENT_LABELS:
        indices = [i for i in range(len(reviews)) if sentimen_labels[i] == label]
        sentiment_result[label] = [
            {"text": reviews[i], "confidence": float(sentimen_scores[i])}
            for i in indices
        ]
        sentiment_wordclouds[label] = generate_wordcloud_words(
            [reviews[i] for i in indices]
        )
        if len(sentimen_labels) > 0:
            sentiment_counts[label] = int(np.sum(sentimen_labels == label))

    for label in EMOTION_LABELS:
        indices = [i for i in range(len(reviews)) if emosi_labels[i] == label]
        emotion_result[label] = [
            {"text": reviews[i], "confidence": float(emosi_scores[i])} for i in indices
        ]
        emotion_wordclouds[label] = generate_wordcloud_words(
            [reviews[i] for i in indices]
        )
        if len(emosi_labels) > 0:
            emotion_counts[label] = int(np.sum(emosi_labels == label))

    # Perbaiki percentages agar total selalu 100
    if len(sentimen_labels) > 0:
        raw_percentages = [
            round(100 * sentiment_counts[label] / len(sentimen_labels))
            for label in SENTIMENT_LABELS
        ]
        diff = 100 - sum(raw_percentages)
        if diff != 0:
            max_idx = np.argmax([sentiment_counts[label] for label in SENTIMENT_LABELS])
            raw_percentages[max_idx] += diff
        sentiment_percentages = {
            label: raw_percentages[i] for i, label in enumerate(SENTIMENT_LABELS)
        }
    if len(emosi_labels) > 0:
        raw_percentages = [
            round(100 * emotion_counts[label] / len(emosi_labels))
            for label in EMOTION_LABELS
        ]
        diff = 100 - sum(raw_percentages)
        if diff != 0:
            max_idx = np.argmax([emotion_counts[label] for label in EMOTION_LABELS])
            raw_percentages[max_idx] += diff
        emotion_percentages = {
            label: raw_percentages[i] for i, label in enumerate(EMOTION_LABELS)
        }

    return_data = {
        "sentiment_analysis": {
            "reviews_by_sentiment": sentiment_result,
            "word_clouds": sentiment_wordclouds,
        },
        "emotion_analysis": {
            "reviews_by_emotion": emotion_result,
            "word_clouds": emotion_wordclouds,
        },
    }
    # Upload ke GCS
    bucket_name = os.getenv("GCS_BUCKET_NAME", "quicktify-storage")
    import uuid

    file_id = str(uuid.uuid4())
    destination_blob_name = f"sentiment_emotion_results/{file_id}.json"
    from app.utils.gcs import upload_json_to_gcs

    file_url = upload_json_to_gcs(return_data, bucket_name, destination_blob_name)
    end_time = time.time()
    logging.info(f"[SentimentEmotion] Waktu proses: {end_time - start_time:.3f} detik")
    return {
        "sentiment_analysis": {
            "percentages": sentiment_percentages,
            "counts": sentiment_counts,
            "reviews_by_sentiment": sentiment_result,
            "word_clouds": sentiment_wordclouds,
        },
        "emotion_analysis": {
            "percentages": emotion_percentages,
            "counts": emotion_counts,
            "reviews_by_emotion": emotion_result,
            "word_clouds": emotion_wordclouds,
        },
        "file_url": file_url,
    }
