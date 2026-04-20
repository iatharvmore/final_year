import streamlit as st
import pandas as pd
import google.generativeai as genai

def run_customer_agent(api_key, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a Customer Experience AI Agent. Review the following recent feedback and support tickets.
    Identify common pain points, categorize feedback sentiment, and propose improvements.
    
    Feedback Data:
    {data}
    
    Format response:
    1. Sentiment Overview
    2. Key Pain Points
    3. Suggested Improvements
    """
    response = model.generate_content(prompt)
    return response.text

def render_customer_agent(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #f2994a 0%, #f2c94c 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1>🤝 Customer Agent Dashboard</h1>
        <p>AI-driven customer experience and support ticket analysis platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("Please configure your Gemini API Key in the global configuration.")
        return

    st.subheader("Recent Customer Feedback")
    
    # Dummy data
    dummy_data = pd.DataFrame({
        "TicketID": ["TCK-101", "TCK-102", "TCK-103", "TCK-104", "TCK-105"],
        "Customer": ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Soylent"],
        "Issue": ["Login timeout error", "Billing discrepancy", "Feature request: Dark mode", "API rate limit exceeded", "App crashes on launch"],
        "Status": ["Open", "Resolved", "Pending", "Open", "Critical"],
        "Sentiment": ["Frustrated", "Neutral", "Positive", "Angry", "Angry"]
    })
    
    st.dataframe(dummy_data, use_container_width=True)
    
    st.divider()
    
    if st.button("💬 Analyze Customer Sentiment", key="customer_analyze_btn"):
        with st.spinner("AI Customer Agent is analyzing..."):
            try:
                insights = run_customer_agent(api_key, dummy_data.to_string())
                st.markdown("### Customer Experience Insights")
                st.markdown(insights)
            except Exception as e:
                st.error(f"Error: {e}")
