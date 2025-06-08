from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    app_rating_prediction_endpoint,
    sentiment_emotion_classification_endpoint,
    spam_detection_endpoint,
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(sentiment_emotion_classification_endpoint.router, prefix="/api")
app.include_router(spam_detection_endpoint.router, prefix="/api")
app.include_router(app_rating_prediction_endpoint.router, prefix="/api")


# Health check endpoint
@app.get("/")
def root():
    return {"message": "API is running"}
