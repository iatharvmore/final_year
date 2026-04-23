import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from TRACKX.data import generate_synthetic_data

def run_agent_orchestration(api_key, data):
    genai.configure(api_key=api_key)
    import os
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    You are TrackX, an enterprise performance intelligence AI agent. 
    Analyze the following holistic performance data spanning HRMS, ERP, and CRM domains. 
    
    Identify key insights, flag anomalies (e.g., low attendance, budget overruns, unmet targets), 
    and provide actionable recommendations tailored to each employee or department.
    
    Data Context:
    {data}
    
    Format your response cleanly using Markdown, with sections for:
    1. Executive Summary
    2. Key Anomalies & Risks
    3. Actionable Recommendations
    """
    response = model.generate_content(prompt)
    return response.text

def render_trackx_agent(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 50%, #a6ffcb 100%); color: #000; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1>TrackX Agent Dashboard</h1>
        <p>Distributed AI agent framework designed to monitor employee performance across enterprise data sources (HRMS, ERP, CRM).</p>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.warning("Please configure your Gemini API Key in the left sidebar to use TrackX.")
        return

    hrms_df, erp_df, crm_df, merged_df = generate_synthetic_data()
    
    st.subheader("Enterprise Data Integrations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Integrated View", "HRMS Data", "ERP Data", "CRM Data"])
    with tab1:
        st.dataframe(merged_df, use_container_width=True)
    with tab2:
        st.dataframe(hrms_df, use_container_width=True)
    with tab3:
        st.dataframe(erp_df, use_container_width=True)
    with tab4:
        st.dataframe(crm_df, use_container_width=True)
    
    st.divider()
    st.subheader("Performance Intelligence Agent")
    
    if st.button("Generate Insights & Recommendations", type="primary", key="trackx_generate_btn"):
        with st.spinner("TrackX Agents are analyzing HRMS, ERP, and CRM data..."):
            try:
                insights = run_agent_orchestration(api_key, merged_df.to_string())
                st.markdown("### Agent Insights")
                st.markdown(insights)
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")
