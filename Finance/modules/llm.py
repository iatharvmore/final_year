import os
from langchain_google_genai import ChatGoogleGenerativeAI
from Finance.modules.agent import run_finance_agent

def generate_summary(df):
    return f"""
    Total Expense: {df['expense'].sum()}
    Total Budget: {df['budget'].sum()}
    Average Expense: {df['expense'].mean()}
    Variance: {df['variance'].sum()}
    Anomalies: {len(df[df['anomaly']=="Yes"])}
    """

def llm_insights(summary):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
    
    prompt = f"Analyze the following financial data summary and provide 3-4 concise, actionable insights:\n\n{summary}"
    response = llm.invoke(prompt)
    return response.content

def chat_with_data(user_query, summary, session_id="default_session", df=None):
    if df is not None:
        # Include up to 100 rows of the dataframe to give the agent granular context
        df_context = f"\n\nDetailed Data (first 100 rows, CSV format):\n{df.head(100).to_csv(index=False)}"
        summary = summary + df_context
        
    return run_finance_agent(session_id, user_query, summary)
