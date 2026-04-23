import streamlit as st
import pandas as pd
import google.generativeai as genai

from Finance.modules.forecasting import forecast
from Finance.modules.anomaly import detect_anomaly
from Finance.modules.variance import variance
from Finance.modules.recommendations import recommend
from Finance.modules.llm import generate_summary, llm_insights, chat_with_data

def render_finance_agent(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #064e3b 0%, #059669 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1>Finance Agent Dashboard</h1>
        <p>Automated financial analysis, variance detection, and intelligent chat.</p>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.warning("Please configure your Gemini API Key in the global configuration.")
        return

    # Configure GenAI
    genai.configure(api_key=api_key)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="fin_uploader")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("Finance/data/sample_finance.csv")
        except FileNotFoundError:
            st.error("Sample dataset not found.")
            return

    df['date'] = pd.to_datetime(df['date'])

    # Create Sub-Tabs
    tab_dashboard, tab_chat = st.tabs(["Dashboard & Analytics", "Chat with Data"])

    with tab_dashboard:
        st.subheader("Data & Trend")
        
        # Use Streamlit's native line chart instead of matplotlib
        chart_data = df.set_index('date')[['expense', 'budget']]
        st.line_chart(chart_data)

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast")
            st.write(forecast(df))
            
            var_df = variance(df.copy())
            an_df = detect_anomaly(var_df.copy())
            
            st.subheader("Alerts")
            st.dataframe(var_df[var_df['variance'].abs() > 1000], use_container_width=True)

        with col2:
            st.subheader("AI Insights")
            summary = generate_summary(an_df)
            if st.button("Generate AI Insights", key="fin_insights_btn"):
                with st.spinner("Generating..."):
                    st.info(llm_insights(summary))
                    
            st.subheader("Recommendations")
            for r in recommend(an_df):
                st.write("• " + r)

    with tab_chat:
        # We still need the summary variable for the chat context if it wasn't defined above yet
        # Since it is defined inside tab_dashboard context, let's redefine or ensure it is available globally.
        var_df_chat = variance(df.copy())
        an_df_chat = detect_anomaly(var_df_chat.copy())
        chat_summary = generate_summary(an_df_chat)

        st.subheader("Finance Chatbot")
        st.write("Ask questions about your financial data, forecasts, and variance alerts.")

        if "fin_chat_history" not in st.session_state:
            st.session_state.fin_chat_history = []

        # Display chat messages
        for message in st.session_state.fin_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about your finances (e.g. why is variance high?)..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.fin_chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = chat_with_data(prompt, chat_summary)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.fin_chat_history.append({"role": "assistant", "content": response})
