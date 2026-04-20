import streamlit as st
import pandas as pd
import google.generativeai as genai

def run_finance_agent(api_key, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""
    You are an AI Financial Forecaster and Analyst. Review the following corporate financial data (Q1 vs Q2).
    Identify areas of budget overrun, forecast next quarter revenue, and provide cost-cutting recommendations.
    
    Financial Data:
    {data}
    
    Format response:
    1. Financial Executive Summary
    2. Variance Analysis (Q1 vs Q2)
    3. Actionable Cost-Saving Measures
    """
    response = model.generate_content(prompt)
    return response.text

def render_finance_agent(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1>💰 Finance Agent Dashboard</h1>
        <p>Automated financial analysis, variance tracking, and cost optimization via AI.</p>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.warning("Please configure your Gemini API Key in the global configuration.")
        return

    st.subheader("Quarterly Financial Overview")
    
    # Dummy data
    dummy_finance_data = pd.DataFrame({
        "Department": ["Engineering", "Sales", "Marketing", "HR", "Operations"],
        "Q1_Budget_$": [500000, 300000, 200000, 100000, 150000],
        "Q1_Actual_$": [480000, 320000, 210000, 95000, 160000],
        "Q2_Budget_$": [520000, 310000, 200000, 105000, 150000],
        "Q2_Actual_$": [550000, 290000, 250000, 100000, 180000],
        "Variance_Trend": ["Over by 30k", "Under by 20k", "Over by 50k", "Under by 5k", "Over by 30k"]
    })
    
    st.dataframe(dummy_finance_data, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Q2 Budget", "$1,285,000", delta="+$25,000 from Q1")
    with col2:
        st.metric("Total Q2 Actual", "$1,370,000", delta="+$85,000 variance", delta_color="inverse")
    
    st.divider()
    
    if st.button("📊 Generate Financial Forecast & Analysis", key="finance_analyze_btn"):
        with st.spinner("Finance Agent is calculating forecast and variances..."):
            try:
                insights = run_finance_agent(api_key, dummy_finance_data.to_string())
                st.markdown("### Financial Analysis & Recommendations")
                st.markdown(insights)
            except Exception as e:
                st.error(f"Error: {e}")
