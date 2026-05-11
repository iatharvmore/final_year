import streamlit as st
import pandas as pd
import google.generativeai as genai
import uuid
import json
import plotly.express as px

from Finance.modules.forecasting import forecast
from Finance.modules.anomaly import detect_anomaly
from Finance.modules.variance import variance
from Finance.modules.recommendations import recommend
from Finance.modules.llm import generate_summary, llm_insights, chat_with_data
from Finance.modules.rag_engine import process_and_index_document

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

    # Standardize column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()

    required_cols = ['date', 'expense', 'budget']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}")
        st.info("Please ensure your CSV has 'date', 'expense', and 'budget' columns.")
        return

    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        st.error(f"Error parsing date column: {e}")
        return

    # Create Sub-Tabs
    tab_dashboard, tab_chat = st.tabs(["Dashboard & Analytics", "Chat with Data"])

    with tab_dashboard:
        st.subheader("Data & Trend")
        
        # Use Plotly for an interactive, easy-to-understand chart
        fig = px.line(df, x='date', y=['expense', 'budget'], 
                      labels={'value': 'Amount ($)', 'date': 'Date', 'variable': 'Category'},
                      title="Expense vs Budget Over Time")
        fig.update_layout(
            hovermode="x unified",
            dragmode="pan" # Prevents the rectangle selection, drag will pan instead
        )
        
        # Show a clean modebar with only zoom in/out/reset
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'scrollZoom': True,
            'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            'displaylogo': False
        })

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast")
            st.dataframe(forecast(df), hide_index=True, use_container_width=True)
            
            var_df = variance(df.copy())
            an_df = detect_anomaly(var_df.copy())
            
            st.subheader("Alerts")
            alerts_shown = 0
            
            # 1. Anomaly Alerts (Not purely budget related, just irregular spending)
            anomalies = an_df[an_df['anomaly'] == 'Yes']
            if not anomalies.empty:
                for _, row in anomalies.head(2).iterrows(): # Show top 2 anomalies
                    date_str = row['date'].strftime('%Y-%m-%d')
                    st.error(f"🚨 Irregular Spending: Unusual expense of **${row['expense']:.2f}** detected on **{date_str}**.")
                    alerts_shown += 1
            
            # 2. Budget Alerts (Mix of Over and Under)
            high_variance = var_df[var_df['variance'].abs() > 1000]
            if not high_variance.empty:
                over_budget = high_variance[high_variance['variance'] > 0].nlargest(2, 'variance')
                under_budget = high_variance[high_variance['variance'] < 0].nsmallest(2, 'variance')
                
                for _, row in over_budget.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    st.warning(f"⚠️ Over Budget: On **{date_str}**, expenses exceeded budget by **${row['variance']:.2f}**.")
                    alerts_shown += 1
                    
                for _, row in under_budget.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    st.info(f"💡 Under Budget: On **{date_str}**, expenses were below budget by **${abs(row['variance']):.2f}**.")
                    alerts_shown += 1
            
            if alerts_shown == 0:
                st.success("No significant variances or anomalies detected.")

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

        if "fin_session_id" not in st.session_state:
            st.session_state.fin_session_id = str(uuid.uuid4())



        if "fin_chat_history" not in st.session_state:
            st.session_state.fin_chat_history = []

        # Display chat messages
        for message in st.session_state.fin_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "structured_data" in message:
                    with st.expander("Agent Thinking & Metadata"):
                        st.json(message["structured_data"])

        # Chat input
        if prompt := st.chat_input("Ask your financial assistant..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.fin_chat_history.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response_obj = chat_with_data(prompt, chat_summary, st.session_state.fin_session_id, df)
                
            with st.chat_message("assistant"):
                st.markdown(response_obj.answer)
                struct_data = response_obj.model_dump()
                with st.expander("Agent Thinking & Metadata"):
                    st.json(struct_data)
                    
            st.session_state.fin_chat_history.append({
                "role": "assistant", 
                "content": response_obj.answer,
                "structured_data": struct_data
            })

