from typing import Any, Dict, Literal

from fastapi import APIRouter, BackgroundTasks, Query

from app.schemas.base import ReviewTextList
from app.services.spam_detection import predict_spam

router = APIRouter()


@router.post("/spam-detection", response_model=Dict[str, Any])
def spam_detection_endpoint(
    data: ReviewTextList,
    background_tasks: BackgroundTasks,
    model_type: Literal["nb", "svm"] = Query(
        "nb", description="Pilih model yang digunakan: 'nb' atau 'svm'"
    ),
):
    result = predict_spam(
        data.reviews, model_type=model_type, background_tasks=background_tasks
    )
    return result
    return result
