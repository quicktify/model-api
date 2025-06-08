# QUICKTIFY MODELS API Documentation

## Base URL

```
http://localhost:8080/api
```

---

## Cara Menjalankan API

### 1. Secara Lokal (dengan virtualenv)

```sh
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 9000
```

### 2. Dengan Docker

```sh
docker build -t quicktify-api .
docker run -p 8080:8080 quicktify-api
```

---

## 1. Spam Detection Endpoint

### URL

```
POST /api/spam-detection
```

### Request Body

```json
{
  "reviews": ["Teks review pertama", "Teks review kedua"]
}
```

### Query Parameters

| Name       | Type   | Default | Description                    |
| ---------- | ------ | ------- | ------------------------------ |
| model_type | string | "nb"    | Pilihan model: "nb" atau "svm" |

defaultnya selalu nb (tidak usah diberi pilihan untuk mengubah)

### Response

```json
{
  "percentages": {
    "genuine_review": 49,
    "irrelevant_content": 36,
    "explicit_spam": 14
  },
  "counts": {
    "genuine_review": 491,
    "irrelevant_content": 368,
    "explicit_spam": 141
  },
  "reviews_by_category": {
    "genuine_review": [{ "text": "Teks review pertama", "confidence": 0.98 }],
    "irrelevant_content": [{ "text": "Teks review kedua", "confidence": 0.95 }],
    "explicit_spam": []
  },
  "file_url": "https://storage.googleapis.com/quicktify-storage/spam_detection_results/xxxx.json"
}
```

---

## 2. Sentiment & Emotion Classification Endpoint

### URL

```
POST /api/sentiment-emotion
```

### Request Body

```json
{
  "reviews": ["Teks review pertama", "Teks review kedua"]
}
```

### Query Parameters

| Name       | Type   | Default   | Description                                                                            |
| ---------- | ------ | --------- | -------------------------------------------------------------------------------------- |
| model_type | string | "nn-only" | Pilihan model: "nn-only", "hybrid-svm", "hybrid-nb", "standalone-svm", "standalone-nb" |

defaultnya selalu nn-only (tidak usah diberi pilihan untuk mengubah)

### Response

```json
{
  "sentiment_analysis": {
    "percentages": {
      "Negative": 40,
      "Neutral": 34,
      "Positive": 25
    },
    "counts": {
      "Negative": 402,
      "Neutral": 343,
      "Positive": 255
    },
    "reviews_by_sentiment": {
      "Negative": [{ "text": "Teks review pertama", "confidence": 0.97 }],
      "Neutral": [{ "text": "Teks review kedua", "confidence": 0.92 }],
      "Positive": []
    },
    "word_clouds": {
      "Negative": [{ "word": "buruk", "count": 2 }],
      "Neutral": [{ "word": "biasa", "count": 1 }],
      "Positive": [{ "word": "bagus", "count": 2 }]
    }
  },
  "emotion_analysis": {
    "percentages": {
      "Anger": 13,
      "Fear": 3,
      "Happy": 25,
      "Love": 0,
      "Neutral": 34,
      "Sad": 22
    },
    "counts": {
      "Anger": 135,
      "Fear": 38,
      "Happy": 256,
      "Love": 3,
      "Neutral": 341,
      "Sad": 227
    },
    "reviews_by_emotion": {
      "Anger": [{ "text": "Teks review pertama", "confidence": 0.95 }],
      "Fear": [],
      "Happy": [],
      "Love": [],
      "Neutral": [{ "text": "Teks review kedua", "confidence": 0.9 }],
      "Sad": []
    },
    "word_clouds": {
      "Anger": { "word": "marah", "count": 2 },
      "Fear": { "word": "takut", "count": 1 },
      "Happy": { "word": "senang", "count": 2 },
      "Love": { "word": "cinta", "count": 1 },
      "Neutral": { "word": "biasa", "count": 1 },
      "Sad": { "word": "sedih", "count": 1 }
    }
  },
  "file_url": "https://storage.googleapis.com/quicktify-storage/sentiment_emotion_results/xxxx.json"
}
```

---

## 3. App Rating Prediction Endpoint

### URL

```
POST /api/app-rating-prediction
```

### Request Body

```json
{
  "category": "GAME",
  "rating_count": 1000,
  "installs": 50000,
  "size": 25.5,
  "content_rating": "Everyone",
  "ad_supported": "Yes",
  "in_app_purchases": "No",
  "editors_choice": "No",
  "app_type": "Free"
}
```

### Query Parameters

| Name         | Type   | Default     | Description                                                             |
| ------------ | ------ | ----------- | ----------------------------------------------------------------------- |
| model_choice | string | "xgb_tuned" | Pilihan model: "xgb_not_tuned", "xgb_tuned", "rf_not_tuned", "rf_tuned" |

> **Catatan:** Default selalu "xgb_tuned". Tidak perlu diubah kecuali ingin eksperimen.

### Response

```json
{
  "predicted_rating": 4.23,
  "model_used": "XGBoost Tuned",
  "confidence_interval": [4.1, 4.4],
  "input_summary": { ... },
  "feature_importance": [
    { "feature": "Installs", "importance": 0.32 },
    { "feature": "Category", "importance": 0.21 }
  ],
  "shap_local": [
    { "feature": "Installs", "shap_value": 0.12, "direction": "positive" },
    { "feature": "Category", "shap_value": -0.08, "direction": "negative" }
  ],
  "shap_plots": {
    "bar_plot_url": "https://storage.googleapis.com/quicktify-storage/app_rating_predictions/bar_plots/bar_xxxx.png",
    "waterfall_plot_url": "https://storage.googleapis.com/quicktify-storage/app_rating_predictions/waterfall_plots/waterfall_xxxx.png",
    "force_plot_url": "https://storage.googleapis.com/quicktify-storage/app_rating_predictions/force_plots/force_xxxx.png"
  }
}
```

- **shap_plots**: Link ke gambar explainability (SHAP) di Google Cloud Storage (public).

---

## Catatan Umum

- Semua request body harus dikirim dalam format JSON.
- Pastikan header `Content-Type: application/json` pada setiap request.
- Semua response berisi link file hasil analisis/gambar di Google Cloud Storage yang bisa diakses publik.
- Contoh response di atas adalah ilustrasi, label dan nilai bisa berbeda tergantung data dan model.
- Untuk deployment di Cloud Run, pastikan port yang digunakan adalah 8080.

---

## Struktur Project (General)

```
models-api/
├── app/
│   ├── api/
│   ├── services/
│   ├── schemas/
│   ├── dict/
│   ├── models/
│   └── utils/
├── main.py
├── requirements.txt
├── Dockerfile
└── README.md
```
