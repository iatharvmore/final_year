import google.generativeai as genai
import os

def generate_summary(df):
    return f"""
    Total Expense: {df['expense'].sum()}
    Total Budget: {df['budget'].sum()}
    Average Expense: {df['expense'].mean()}
    Variance: {df['variance'].sum()}
    Anomalies: {len(df[df['anomaly']=="Yes"])}
    """

def llm_insights(summary):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    prompt = f"Analyze financial data and give insights:\n{summary}"
    response = model.generate_content(prompt)
    return response.text

def chat_with_data(user_query, summary):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    You are a financial advisor.
    Data:
    {summary}

    User Question:
    {user_query}
    """
    response = model.generate_content(prompt)
    return response.text
