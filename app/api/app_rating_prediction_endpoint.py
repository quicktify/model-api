from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Query

from app.schemas.base import AppRatingPredictionRequest, AppRatingPredictionResponse
from app.services.app_rating_prediction import predict_app_rating

router = APIRouter()


@router.post("/app-rating-prediction", response_model=AppRatingPredictionResponse)
def app_rating_prediction(
    data: AppRatingPredictionRequest,
    background_tasks: BackgroundTasks,
    model_choice: Literal[
        "xgb_not_tuned", "xgb_tuned", "rf_not_tuned", "rf_tuned"
    ] = Query(
        "xgb_tuned",
        description="Pilih model yang digunakan",
    ),
):
    return predict_app_rating(
        data, model_choice=model_choice, background_tasks=background_tasks
    )
