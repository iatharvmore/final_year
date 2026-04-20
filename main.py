import streamlit as st
import os
from dotenv import load_dotenv

from HR.app import render_hr_agent
from TRACKX.app import render_trackx_agent
from Customer.app import render_customer_agent
from Finance.app import render_finance_agent

def render_summary():
    st.markdown("""
    <div style="padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 2rem; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.2);">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🏢 Enterprise AI Orchestration Platform</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Unified Intelligence for the Pillars of Your Business</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🌟 Welcome to the Multi-LLM Enterprise Orchestration Dashboard
    
    This command center provides AI-driven insights across the four core pillars of your organization.
    Select a tab above to drill down into specific departmental intelligence powered by Generative AI.
    """)
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("### 🧑‍💼 HR\nResume parsing & candidate matching.")
    with col2:
        st.success("### 📈 TrackX\nEmployee performance & integration.")
    with col3:
        st.warning("### 🤝 Customer\nSentiment analysis & feedback.")
    with col4:
        st.error("### 💰 Finance\nBudget variance & forecasting.")

def main():
    st.set_page_config(page_title="Enterprise OS", layout="wide", page_icon="🏢")
    load_dotenv()
    
    with st.sidebar:
        st.title("⚙️ Global Configuration")
        global_api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""), help="Required for all agents. Will be applied universally.")
        
        if global_api_key:
            os.environ["GOOGLE_API_KEY"] = global_api_key
            os.environ["GEMINI_API_KEY"] = global_api_key
            st.success("✅ API Key Active")
        else:
            st.warning("⚠️ Please provide an API key.")
        
        st.divider()
        st.markdown("### Agent Status")
        st.markdown("- 🟢 HR Agent")
        st.markdown("- 🟢 TrackX Agent")
        st.markdown("- 🟢 Customer Agent")
        st.markdown("- 🟢 Finance Agent")
    
    # 5 Tabs
    t1, t2, t3, t4, t5 = st.tabs(["📊 Whole Summary", "🧑‍💼 HR Agent", "📈 TrackX Agent", "🤝 Customer Agent", "💰 Finance Agent"])
    
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
