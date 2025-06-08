from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ReviewTextList(BaseModel):
    reviews: List[str]


class ReviewItem(BaseModel):
    text: str
    date: Optional[str] = None
    user: Optional[str] = None
    rating: Optional[int] = None


class ReviewObjectList(BaseModel):
    reviews: List[ReviewItem]


CATEGORY_CHOICES = [
    "Adventure",
    "Tools",
    "Productivity",
    "Communication",
    "Social",
    "Libraries & Demo",
    "Lifestyle",
    "Personalization",
    "Racing",
    "Maps & Navigation",
    "Travel & Local",
    "Food & Drink",
    "Books & Reference",
    "Medical",
    "Puzzle",
    "Entertainment",
    "Arcade",
    "Auto & Vehicles",
    "Photography",
    "Health & Fitness",
    "Education",
    "Shopping",
    "Board",
    "Music & Audio",
    "Sports",
    "Beauty",
    "Business",
    "Educational",
    "Finance",
    "News & Magazines",
    "Casual",
    "Art & Design",
    "House & Home",
    "Card",
    "Events",
    "Trivia",
    "Weather",
    "Strategy",
    "Word",
    "Video Players & Editors",
    "Action",
    "Simulation",
    "Music",
    "Dating",
    "Role Playing",
    "Casino",
    "Comics",
    "Parenting",
]
CONTENT_RATING_CHOICES = [
    "Everyone",
    "Teen",
    "Mature 17+",
    "Everyone 10+",
    "Adults only 18+",
    "Unrated",
]
APP_TYPE_CHOICES = ["Free", "Paid"]


class AppRatingPredictionRequest(BaseModel):
    category: str = Field(..., description="Kategori aplikasi", example="GAME")
    rating_count: int = Field(
        ..., ge=0, le=56025424, description="Jumlah rating", example=12345
    )
    installs: int = Field(
        ..., ge=0, le=1000000000, description="Jumlah installs", example=1000000
    )
    size: float = Field(
        ..., ge=0, le=1500, description="Ukuran aplikasi (MB)", example=25.5
    )
    content_rating: str = Field(..., description="Content rating", example="Everyone")
    ad_supported: bool = Field(..., description="Apakah aplikasi mendukung iklan")
    in_app_purchases: bool = Field(
        ..., description="Apakah aplikasi memiliki in-app purchases"
    )
    editors_choice: bool = Field(..., description="Apakah aplikasi editors choice")
    app_type: str = Field(..., description="Tipe aplikasi", example="Free")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        if v not in CATEGORY_CHOICES:
            raise ValueError(f"category harus salah satu dari: {CATEGORY_CHOICES}")
        return v

    @field_validator("content_rating")
    @classmethod
    def validate_content_rating(cls, v):
        if v not in CONTENT_RATING_CHOICES:
            raise ValueError(
                f"content_rating harus salah satu dari: {CONTENT_RATING_CHOICES}"
            )
        return v

    @field_validator("app_type")
    @classmethod
    def validate_app_type(cls, v):
        if v not in APP_TYPE_CHOICES:
            raise ValueError(f"app_type harus salah satu dari: {APP_TYPE_CHOICES}")
        return v


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ShapLocalItem(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ShapPlots(BaseModel):
    bar_plot_url: str
    waterfall_plot_url: str
    force_plot_url: str


class AppRatingPredictionResponse(BaseModel):
    predicted_rating: float
    model_used: str
    confidence_interval: Optional[List[float]] = None
    input_summary: dict
    feature_importance: Optional[List[FeatureImportanceItem]] = None
    shap_local: Optional[List[ShapLocalItem]] = None
    shap_plots: Optional[ShapPlots] = None