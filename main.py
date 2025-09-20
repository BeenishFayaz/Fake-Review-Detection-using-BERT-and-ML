from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sentence_transformers import SentenceTransformer


# Load trained model + BERT

with open("trained_fakereview_model.pkl", "rb") as f:
    model = pickle.load(f)

bert_model = SentenceTransformer("bert-base-nli-mean-tokens")


# FastAPI app
# Create App Object
app = FastAPI(title="Fake Review Detection API",
              description="Detect whether a review is Truthful or Deceptive",
              version="1.0")


# Input schema

class ReviewInput(BaseModel):
    review: str


# Prediction endpoint

@app.post("/predict")
def predict_review(data: ReviewInput):
    # Step 1: Generate embeddings
    embeddings = bert_model.encode([data.review])

    # Step 2: Predict with trained model
    prediction = model.predict(embeddings)[0]

    # Step 3: Map prediction to label
    result = "Truthful" if prediction == 0 else "Deceptive"

    return {"review": data.review, "prediction": result}




# uvicorn is the server that runs your FastAPI app.
# By default, it starts on localhost (127.0.0.1) and port 8000.
# uvicorn main:app --reload