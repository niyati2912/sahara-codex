# main.py
import json
from transformers import pipeline
from fastapi import FastAPI, Body
from pydantic import BaseModel

# Create a data model for the incoming request
# This tells FastAPI what kind of data to expect.
class TriageRequest(BaseModel):
    text: str

# Create the FastAPI app
app = FastAPI()

# Load the trained model from the folder you created.
# This is your "Detector" AI.
print("Loading AI model...")
model_path = "./sahara-medical-model"
detector = pipeline("text-classification", model=model_path, tokenizer=model_path)
print("Model loaded.")

# Load the first-aid knowledge base you created.
# This is your "Advisor".
print("Loading First-Aid Knowledge Base...")
with open("first_aid_kb.json", "r") as f:
    knowledge_base = json.load(f)
print("Knowledge Base loaded.")


# Define the main API endpoint
@app.post("/triage")
def triage_emergency(request: TriageRequest):
    """
    Accepts a user's emergency text, predicts the medical condition,
    and returns the appropriate first-aid steps.
    """
    # Get the user's message from the request
    symptom_text = request.text
    print(f"Received symptom text: {symptom_text}")

    # Use the AI model to predict the condition
    prediction = detector(symptom_text)[0]
    condition = prediction['label']
    confidence = prediction['score']
    
    print(f"Predicted Condition: {condition} with confidence {confidence:.2f}")

    # Use the predicted condition to get the first-aid steps
    # If the condition is not in our knowledge base, use the "Default" steps.
    first_aid_steps = knowledge_base.get(condition, knowledge_base["Default"])

    # Return the final JSON response
    return {
        "detected_condition": condition,
        "confidence_score": confidence,
        "first_aid_protocol": first_aid_steps
    }

# A simple root endpoint to confirm the server is running
@app.get("/")
def read_root():
    return {"message": "Sahara Medical AI Triage Server is running."}