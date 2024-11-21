import streamlit as st
import google.generativeai as genai
from langdetect import detect
from googletrans import Translator
from io import BytesIO
from fpdf import FPDF
import concurrent.futures
import json
import time
from textblob import TextBlob
import pandas as pd
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure API Key securely from Streamlit's secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# App Configuration
st.set_page_config(page_title="Fast Email Storytelling AI", page_icon="", layout="wide")
st.title("Lightning-Fast Email Storytelling AI")
st.write("Rapidly extract insights and generate professional responses from emails.")

# Sidebar for Features
st.sidebar.header("Settings")
features = {
    "sentiment": st.sidebar.checkbox("Perform Sentiment Analysis"),
    "highlights": st.sidebar.checkbox("Highlight Key Phrases"),
    "response": st.sidebar.checkbox("Generate Suggested Response"),
    "export": st.sidebar.checkbox("Export Options"),
    "tone_analysis": st.sidebar.checkbox("Analyze Email Tone"),
    "translate": st.sidebar.checkbox("Translate Email"),
    "follow_up": st.sidebar.checkbox("Set Follow-up Reminder"),
    "emotional_impact": st.sidebar.checkbox("Analyze Emotional Impact"),
    "signature": st.sidebar.checkbox("Generate Email Signature")
}

# Input Email Section
email_content = st.text_area("Paste your email content here:", height=200)

MAX_EMAIL_LENGTH = 1000

# Cache the AI responses to improve performance
@st.cache_data(ttl=3600)
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

# Sentiment Analysis
def get_sentiment(email_content):
    return TextBlob(email_content).sentiment.polarity

# Tone Analysis using Hugging Face
tone_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_tone(text):
    candidate_labels = ["Formal", "Friendly", "Aggressive", "Neutral"]
    return tone_analyzer(text, candidate_labels)

# PDF Export
def export_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    return pdf.output(dest='S').encode('latin1')

# CSV Export
def export_csv(text, response, highlights, sentiment):
    df = pd.DataFrame({
        "Summary": [text],
        "Suggested Response": [response],
        "Highlights": [highlights],
        "Sentiment": [sentiment],
    })
    return df.to_csv(index=False).encode('utf-8')

# Word Cloud Visualization
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt

# Emotional Impact Scoring
def emotional_impact_score(text):
    positive_words = ["happy", "great", "excited"]
    negative_words = ["angry", "stress", "disappointed"]
    score = 0

    for word in positive_words:
        if word in text.lower():
            score += 1
    for word in negative_words:
        if word in text.lower():
            score -= 1
    return score

# Translate to Preferred Language
def translate_to_preferred_language(text, target_lang="en"):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

# Follow-up Reminder System
follow_up_time = timedelta(days=3)  # Automatically remind after 3 days

def set_follow_up(email_timestamp):
    follow_up_date = email_timestamp + follow_up_time
    return follow_up_date

# Personalized Email Response Generator
def generate_personalized_response(email_content, user_style="formal"):
    prompt = f"Generate a {user_style} response to this email:\n\n{email_content}"
    return get_ai_response(prompt, email_content)

if email_content and st.button("Generate Insights"):
    try:
        detected_lang = detect(email_content)
        user_language = st.sidebar.selectbox("Select your preferred language:", ["en", "es", "fr", "de", "it", "pt"])

        # Check if translation is needed
        if features["translate"] and detected_lang != user_language:
            translated_email = translate_to_preferred_language(email_content, target_lang=user_language)
            st.subheader(f"Translated Email ({user_language})")
            st.write(translated_email)
        else:
            translated_email = email_content

        with st.spinner("Generating insights..."):
            progress_bar = st.progress(0)

            # Using ThreadPoolExecutor for concurrent tasks
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_summary = executor.submit(get_ai_response, "Summarize the email in a concise, actionable format:\n\n", translated_email)
                future_response = executor.submit(get_ai_response, "Draft a professional response to this email:\n\n", translated_email) if features["response"] else None
                future_highlights = executor.submit(get_ai_response, "Highlight key points and actions in this email:\n\n", translated_email) if features["highlights"] else None
                future_tone = executor.submit(analyze_tone, translated_email) if features["tone_analysis"] else None
                future_emotional_impact = executor.submit(emotional_impact_score, translated_email) if features["emotional_impact"] else None

                summary = future_summary.result()
                response = future_response.result() if future_response else ""
                highlights = future_highlights.result() if future_highlights else ""
                tone = future_tone.result() if future_tone else None
                emotional_impact = future_emotional_impact.result() if future_emotional_impact else None

                # Update progress bar
                progress_bar.progress(50)

            # Displaying Results
            st.subheader("AI Summary")
            st.write(summary)

            if features["response"]:
                st.subheader("Suggested Response")
                st.write(response)

            if features["highlights"]:
                st.subheader("Key Highlights")
                st.write(highlights)

            if features["sentiment"]:
                sentiment = get_sentiment(translated_email)
                sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
                st.subheader("Sentiment Analysis")
                st.write(f"**Sentiment:** {sentiment_label} (Polarity: {sentiment:.2f})")

            if features["tone_analysis"]:
                st.subheader("Tone Analysis")
                st.write(f"Detected Tone: {tone['labels'][0]} (Confidence: {tone['scores'][0]:.2f})")

            if features["emotional_impact"]:
                st.subheader("Emotional Impact")
                st.write(f"Impact Score: {emotional_impact}")

            if features["follow_up"]:
                # Example email timestamp
                email_timestamp = datetime.now()
                follow_up_date = set_follow_up(email_timestamp)
                st.subheader("Follow-up Reminder")
                st.write(f"Next follow-up reminder: {follow_up_date.strftime('%Y-%m-%d %H:%M:%S')}")

            if features["signature"]:
                user_name = st.text_input("Your Name:")
                user_position = st.text_input("Your Position:")
                company_name = st.text_input("Company Name:")

                if user_name and user_position and company_name:
                    email_signature = generate_personalized_response(translated_email, "formal")
                    st.subheader("Generated Email Signature")
                    st.write(email_signature)

            # Update progress bar
            progress_bar.progress(100)

            # Export Options
            if features["export"]:
                export_text = f"Summary:\n{summary}\n\nResponse:\n{response}\n\nHighlights:\n{highlights}"
                export_json = {
                    "summary": summary,
                    "response": response,
                    "highlights": highlights,
                    "sentiment": sentiment_label if features["sentiment"] else None,
                    "tone": tone['labels'][0] if tone else None,
                    "emotional_impact": emotional_impact if emotional_impact else None,
                }

                pdf_buffer = BytesIO(export_pdf(export_text))
                buffer_txt = BytesIO(export_text.encode("utf-8"))
                buffer_json = BytesIO(json.dumps(export_json, indent=4).encode("utf-8"))
                buffer_csv = BytesIO(export_csv(summary, response, highlights, sentiment_label))

                st.download_button("Download as Text", data=buffer_txt, file_name="analysis.txt", mime="text/plain")
                st.download_button("Download as JSON", data=buffer_json, file_name="analysis.json", mime="application/json")
                st.download_button("Download as PDF", data=pdf_buffer, file_name="analysis.pdf", mime="application/pdf")
                st.download_button("Download as CSV", data=buffer_csv, file_name="analysis.csv", mime="text/csv")

            # Word Cloud Visualization
            if features["highlights"]:
                wordcloud = generate_word_cloud(translated_email)
                st.subheader("Word Cloud of Email Content")
                st.pyplot(wordcloud)

    except Exception as e:
        st.error(f"An error occurred: {e}")
