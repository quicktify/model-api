from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Query

from app.schemas.base import ReviewTextList
from app.services.sentiment_emotion_classification import classify_sentiment_emotion

router = APIRouter()


@router.post("/sentiment-emotion")
def analyze_sentiment_emotion(
    data: ReviewTextList,
    background_tasks: BackgroundTasks,
    model_type: Literal[
        "nn-only", "hybrid-svm", "hybrid-nb", "standalone-svm", "standalone-nb"
    ] = Query("nn-only", description="Pilih model yang digunakan"),
):
    result = classify_sentiment_emotion(
        data.reviews, model_type=model_type, background_tasks=background_tasks
    )
    return result