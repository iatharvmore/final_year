import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv

from HR.app import render_hr_agent
from TRACKX.app import render_trackx_agent
from Customer.app import render_customer_agent
from Finance.app import render_finance_agent

def gather_agent_data():
    st.subheader("Enterprise Data Overview")
    st.write("Centralized view of data from all departmental agents.")
    
    col1, col2 = st.columns(2)
    
    # HR Data
    with col1:
        st.markdown("#### Human Resources")
        hr_path = "HR/data/resume_analysis.xlsx"
        if os.path.exists(hr_path):
            try:
                hr_df = pd.read_excel(hr_path)
                st.metric("Processed Resumes", len(hr_df))
                st.dataframe(hr_df.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load HR data: {e}")
        elif os.path.exists("resume_analysis.xlsx"):
            try:
                hr_df = pd.read_excel("resume_analysis.xlsx")
                st.metric("Processed Resumes", len(hr_df))
                st.dataframe(hr_df.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load HR data: {e}")
        else:
            st.info("No HR data available.")

    # Finance Data
    with col2:
        st.markdown("#### Finance")
        fin_path = "Finance/data/sample_finance.csv"
        if os.path.exists(fin_path):
            try:
                fin_df = pd.read_csv(fin_path)
                st.metric("Financial Records", len(fin_df))
                st.dataframe(fin_df.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load Finance data: {e}")
        else:
            st.info("No Finance data available.")

    col3, col4 = st.columns(2)

    # Customer Data
    with col3:
        st.markdown("#### Customer Experience")
        cust_path = "Customer/data/knowledge_base.json"
        if os.path.exists(cust_path):
            try:
                with open(cust_path, "r") as f:
                    kb = json.load(f)
                kb_entries = len(kb.get("knowledge_base", []))
                st.metric("Knowledge Base Entries", kb_entries)
                st.json(kb, expanded=False)
            except Exception as e:
                st.error(f"Could not load Customer data: {e}")
        elif os.path.exists("Customer/knowledge_base.json"):
            try:
                with open("Customer/knowledge_base.json", "r") as f:
                    kb = json.load(f)
                kb_entries = len(kb.get("knowledge_base", []))
                st.metric("Knowledge Base Entries", kb_entries)
                st.json(kb, expanded=False)
            except Exception as e:
                st.error(f"Could not load Customer data: {e}")
        else:
            st.info("No Customer data available.")

    # TrackX Data
    with col4:
        st.markdown("#### Performance Integrations (TrackX)")
        trackx_path = "TRACKX/data/synthetic_data.csv"
        if os.path.exists(trackx_path):
            try:
                tx_df = pd.read_csv(trackx_path)
                st.metric("Performance Records", len(tx_df))
                st.dataframe(tx_df.head(3), use_container_width=True)
            except Exception as e:
                st.error(f"Could not load TrackX data: {e}")
        else:
            st.info("Performance data is generated dynamically or not yet saved.")

def render_summary():
    st.markdown("""
    <div style="padding: 3rem; background: #1e293b; color: #f8fafc; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3b82f6;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 600; letter-spacing: -0.025em;">Enterprise AI Orchestration Platform</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; font-weight: 400;">Unified Intelligence for Organizational Pillars</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Executive Dashboard
    
    This command center provides AI-driven insights across the four core pillars of the organization.
    Select a module above to access departmental intelligence powered by Generative AI.
    """)
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("#### Human Resources\nResume parsing and candidate matching algorithms.")
    with col2:
        st.success("#### Performance (TrackX)\nEmployee performance and integration analytics.")
    with col3:
        st.warning("#### Customer Experience\nSentiment analysis and feedback processing.")
    with col4:
        st.error("#### Financial Analytics\nBudget variance and predictive forecasting.")

def main():
    st.set_page_config(page_title="Enterprise OS", layout="wide")
    load_dotenv()
    
    global_api_key = os.getenv("GOOGLE_API_KEY", "")
    os.environ["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    with st.sidebar:
        st.title("System Configuration")
        
        if global_api_key:
            os.environ["GEMINI_API_KEY"] = global_api_key
            st.success("System Status: Operational")
        else:
            st.error("System Status: Configuration Required (.env)")
        
        st.divider()
        st.markdown("### Module Status")
        st.markdown("- Human Resources: Active")
        st.markdown("- Performance (TrackX): Active")
        st.markdown("- Customer Experience: Active")
        st.markdown("- Financial Analytics: Active")
        
        st.divider()
        st.markdown(f"**Model in use:** {os.environ['GEMINI_MODEL']}")
    
    # 5 Tabs
    t1, t2, t3, t4, t5 = st.tabs(["Executive Summary", "Human Resources", "Performance Integrations", "Customer Experience", "Financial Analytics"])
    
    with t1:
        render_summary()
    with t2:
        render_hr_agent(api_key=global_api_key)
    with t3:
        render_trackx_agent(api_key=global_api_key)
    with t4:
        render_customer_agent(api_key=global_api_key)
    with t5:
        render_finance_agent(api_key=global_api_key)

if __name__ == "__main__":
    main()
