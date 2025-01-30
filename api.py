from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai

# Initialize FastAPI
app = FastAPI()

# Configure Google API key
genai.configure(api_key="your_google_api_key_here")

# Define Request Model
class EmailRequest(BaseModel):
    email_content: str
    features: dict

# Define Response Model
class EmailResponse(BaseModel):
    summary: str
    sentiment: str
    response: str | None
    key_phrases: list[str]

# AI Function
def get_ai_response(prompt, content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + content)
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

# Sentiment Analysis
def analyze_sentiment(text):
    positive_words = ["good", "great", "excellent", "happy"]
    negative_words = ["bad", "sad", "angry", "poor"]
    score = sum([1 for word in text.split() if word in positive_words]) - sum([1 for word in text.split() if word in negative_words])
    return "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"

# Key Phrase Extraction
def extract_key_phrases(text):
    import re
    return list(set(re.findall(r"\b[A-Za-z]{4,}\b", text)))

# API Endpoint
@app.post("/analyze/", response_model=EmailResponse)
def analyze_email(request: EmailRequest):
    email_content = request.email_content
    features = request.features

    # AI-based Summary
    summary = get_ai_response("Summarize this email:\n\n", email_content) if features.get("summary") else ""

    # AI-generated Response
    response = get_ai_response("Draft a response:\n\n", email_content) if features.get("response") else ""

    # Sentiment Analysis
    sentiment = analyze_sentiment(email_content) if features.get("sentiment") else "Not Analyzed"

    # Key Phrase Extraction
    key_phrases = extract_key_phrases(email_content) if features.get("key_phrases") else []

    return {"summary": summary, "sentiment": sentiment, "response": response, "key_phrases": key_phrases}

# Root Endpoint
@app.get("/")
def home():
    return {"message": "FastAPI is running on Render!"}
