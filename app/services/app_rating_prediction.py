import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from app.schemas.base import (
    AppRatingPredictionRequest,
    AppRatingPredictionResponse,
    FeatureImportanceItem,
    ShapPlots,
)
from app.utils.gcs import upload_image_to_gcs

matplotlib.use("Agg")


# Load artifacts once (singleton pattern)
ARTIFACTS_DIR = "app/models/app_rating_prediction_artifacts"

# Helper to load artifact
_DEF = object()


def _load_artifact(name, default=_DEF):
    path = os.path.join(ARTIFACTS_DIR, name)
    if os.path.exists(path):
        return joblib.load(path)
    if default is not _DEF:
        return default
    raise FileNotFoundError(f"Artifact not found: {path}")


# Load all mappings
category_mapping = _load_artifact("category_mapping.pkl")
reverse_category_mapping = _load_artifact("reverse_category_mapping.pkl")
content_rating_mapping = _load_artifact("content_rating_mapping.pkl")
reverse_content_rating_mapping = _load_artifact("reverse_content_rating_mapping.pkl")
ad_supported_mapping = _load_artifact("ad_supported_custom_mapping.pkl")
ad_supported_reverse_mapping = _load_artifact("ad_supported_reverse_mapping.pkl")
in_app_purchases_mapping = _load_artifact("in_app_purchases_custom_mapping.pkl")
in_app_purchases_reverse_mapping = _load_artifact(
    "in_app_purchases_reverse_mapping.pkl"
)
editors_choice_mapping = _load_artifact("editors_choice_custom_mapping.pkl")
editors_choice_reverse_mapping = _load_artifact("editors_choice_reverse_mapping.pkl")
app_type_mapping = _load_artifact("app_type_custom_mapping.pkl")
app_type_reverse_mapping = _load_artifact("app_type_reverse_mapping.pkl")
scaler = _load_artifact("standard_scaler.pkl")


# Load XGBoost models from .json (native format)
def load_xgb_model(bin_path):
    model = xgb.XGBRegressor()
    model.load_model(bin_path)
    return model


# Load models
xgb_not_tuned = load_xgb_model(
    os.path.join(ARTIFACTS_DIR, "regressor_xgb_not_tuned.ubj")
)
xgb_tuned = load_xgb_model(os.path.join(ARTIFACTS_DIR, "regressor_xgb_tuned.ubj"))
rf_not_tuned = _load_artifact("regressor_rf_not_tuned.pkl")
rf_tuned = _load_artifact("regressor_rf_tuned.pkl")


MODEL_OPTIONS = {
    "xgb_not_tuned": (xgb_not_tuned, "XGBoost Not Tuned"),
    "xgb_tuned": (xgb_tuned, "XGBoost Tuned"),
    "rf_not_tuned": (rf_not_tuned, "RandomForest Not Tuned"),
    "rf_tuned": (rf_tuned, "RandomForest Tuned"),
}


def preprocess_input(data: AppRatingPredictionRequest) -> np.ndarray:
    feature_names = [
        "Category",
        "Rating Count",
        "Installs",
        "Size",
        "Content Rating",
        "Ad Supported",
        "In App Purchases",
        "Editors Choice",
        "App Type",
    ]
    features = [
        category_mapping.get(data.category, 0),
        data.rating_count,
        data.installs,
        data.size,
        content_rating_mapping.get(data.content_rating, 0),
        ad_supported_mapping.get(data.ad_supported, 0),
        in_app_purchases_mapping.get(data.in_app_purchases, 0),
        editors_choice_mapping.get(data.editors_choice, 0),
        app_type_mapping.get(data.app_type, 0),
    ]
    df = pd.DataFrame([features], columns=feature_names)
    arr_scaled = scaler.transform(df)
    return arr_scaled


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        return [
            FeatureImportanceItem(feature=f, importance=float(i))
            for f, i in zip(feature_names, importances)
        ]
    return None


def get_confidence_interval(model, arr_scaled):
    # Only for RF: use std dev of predictions from all trees
    if hasattr(model, "estimators_"):
        preds = np.array([tree.predict(arr_scaled)[0] for tree in model.estimators_])
        mean = float(np.mean(preds))
        std = float(np.std(preds))
        return [round(mean - std, 2), round(mean + std, 2)]
    return None


# Global cache untuk SHAP explainer
SHAP_EXPLAINER_CACHE = {}


def get_shap_explainer(model):
    if model in SHAP_EXPLAINER_CACHE:
        return SHAP_EXPLAINER_CACHE[model]
    explainer = shap.TreeExplainer(model)
    SHAP_EXPLAINER_CACHE[model] = explainer
    return explainer


# Async GCS upload helper
_executor = ThreadPoolExecutor(max_workers=2)


def async_upload_image_to_gcs(file_path, bucket_name, dest_path):
    def _upload():
        upload_image_to_gcs(file_path, bucket_name, dest_path)

    _executor.submit(_upload)


def generate_and_upload_shap_plots(
    model,
    explainer,
    arr_scaled,
    feature_names,
    bucket_name,
    base_folder,
    unique_id,
    background_tasks=None,
):
    shap_values = explainer.shap_values(arr_scaled)
    explanation = shap.Explanation(
        values=shap_values[0]
        if hasattr(shap_values, "__len__") and len(shap_values) > 0
        else shap_values,
        base_values=explainer.expected_value,
        data=arr_scaled[0],
        feature_names=feature_names,
    )
    plot_urls = {}
    # Bar Plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_bar:
        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig(tmp_bar.name)
        plt.close()
        bar_dest = f"{base_folder}/bar_plots/bar_{unique_id}.png"
        bar_url = upload_image_to_gcs(tmp_bar.name, bucket_name, bar_dest)
        plot_urls["bar_plot_url"] = bar_url
    # Waterfall Plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_waterfall:
        shap.plots.waterfall(explanation, show=False, max_display=len(feature_names))
        plt.tight_layout()
        plt.savefig(tmp_waterfall.name)
        plt.close()
        waterfall_dest = f"{base_folder}/waterfall_plots/waterfall_{unique_id}.png"
        waterfall_url = upload_image_to_gcs(
            tmp_waterfall.name, bucket_name, waterfall_dest
        )
        plot_urls["waterfall_plot_url"] = waterfall_url
    # Force Plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_force:
        shap.force_plot(
            explanation.base_values,
            explanation.values,
            explanation.data,
            feature_names=explanation.feature_names,
            matplotlib=True,
            show=False,
            figsize=(30, 8),
        )
        plt.tight_layout()
        plt.savefig(tmp_force.name)
        plt.close()
        force_dest = f"{base_folder}/force_plots/force_{unique_id}.png"
        force_url = upload_image_to_gcs(tmp_force.name, bucket_name, force_dest)
        plot_urls["force_plot_url"] = force_url
    return plot_urls, shap_values[0] if hasattr(shap_values, "__len__") and len(
        shap_values
    ) > 0 else shap_values


def generate_recommendations(shap_local):
    sorted_feats = sorted(shap_local, key=lambda x: abs(x["shap_value"]), reverse=True)
    recs = []
    for feat in sorted_feats[:2]:
        if feat["direction"] == "negative":
            recs.append(
                f"Pertimbangkan untuk memperbaiki atau mengoptimalkan fitur {feat['feature']}."
            )
        else:
            recs.append(
                f"Fitur {feat['feature']} sangat membantu rating, pertahankan atau tingkatkan."
            )
    return recs


def generate_dynamic_insight(pred, shap_local):
    if pred >= 4.5:
        return "Prediksi rating sangat tinggi. Fitur aplikasi Anda sudah sangat baik!"
    elif pred >= 4.0:
        neg_feats = [f["feature"] for f in shap_local if f["direction"] == "negative"]
        if neg_feats:
            return f"Rating tinggi, namun fitur {', '.join(neg_feats)} menurunkan rating. Perbaiki fitur tersebut."
        return "Rating tinggi. Pastikan fitur utama tetap dipertahankan."
    elif pred >= 3.0:
        neg_feats = [f["feature"] for f in shap_local if f["direction"] == "negative"]
        if neg_feats:
            return f"Rating cukup, namun fitur {', '.join(neg_feats)} menurunkan rating. Perbaiki fitur tersebut."
        return "Rating cukup, namun masih ada ruang untuk perbaikan."
    else:
        return (
            "Prediksi rating rendah. Perlu evaluasi fitur utama dan kualitas aplikasi."
        )


def predict_app_rating(
    data: AppRatingPredictionRequest,
    model_choice: str = "xgb_tuned",
    background_tasks=None,
) -> AppRatingPredictionResponse:
    import logging

    start_time = time.time()
    feature_names = [
        "Category",
        "Rating Count",
        "Installs",
        "Size",
        "Content Rating",
        "Ad Supported",
        "In App Purchases",
        "Editors Choice",
        "App Type",
    ]
    arr_scaled = preprocess_input(data)
    model, model_label = MODEL_OPTIONS.get(model_choice, (xgb_tuned, "XGBoost Tuned"))
    pred = float(model.predict(arr_scaled)[0])
    feature_importance = get_feature_importance(model, feature_names)
    confidence_interval = get_confidence_interval(model, arr_scaled)
    input_summary = data.model_dump()

    # SHAP explainability
    explainer = get_shap_explainer(model)
    unique_id = str(uuid.uuid4())
    bucket_name = "quicktify-storage"  # Ganti dengan nama bucket GCS Anda
    base_folder = "app_rating_predictions"
    plot_urls, shap_values = generate_and_upload_shap_plots(
        model,
        explainer,
        arr_scaled,
        feature_names,
        bucket_name,
        base_folder,
        unique_id,
        background_tasks=background_tasks,
    )

    # SHAP local values
    shap_local = []
    for i, val in enumerate(shap_values):
        direction = "positive" if val >= 0 else "negative"
        shap_local.append(
            {
                "feature": feature_names[i],
                "shap_value": float(val),
                "direction": direction,
            }
        )

    shap_plots_obj = ShapPlots(**plot_urls)

    end_time = time.time()
    logging.info(
        f"[AppRatingPrediction] Waktu proses: {end_time - start_time:.3f} detik"
    )

    return AppRatingPredictionResponse(
        predicted_rating=round(pred, 3),
        model_used=model_label,
        confidence_interval=confidence_interval,
        input_summary=input_summary,
        feature_importance=feature_importance,
        shap_local=shap_local,
        shap_plots=shap_plots_obj,
    )
