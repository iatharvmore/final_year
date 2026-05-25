import builtins
import types
import sys

# Monkeypatch builtins.eval to prevent class-level methods (like Chain.dict) from shadowing
# built-in type names (like dict) during Pydantic/typing annotation evaluation in Python 3.14+.
# We use sys._getframe(1) to correctly resolve the caller's globals and locals when not explicitly passed,
# preserving correct standard library/third-party evaluations (like numpy.f2py).
_original_eval = builtins.eval

def _patched_eval(code, globals=None, locals=None):
    if globals is None:
        try:
            frame = sys._getframe(1)
            globals = frame.f_globals
            locals = frame.f_locals
        except Exception:
            pass
    elif locals is None:
        locals = globals

    if locals is not None:
        _shadowed_builtins = {'dict', 'list', 'set', 'tuple', 'str', 'int', 'float', 'bool', 'type'}
        _has_shadowed = False
        for _name in _shadowed_builtins:
            try:
                if _name in locals and isinstance(locals[_name], (types.FunctionType, types.MethodType)):
                    _has_shadowed = True
                    break
            except Exception:
                pass
        
        if _has_shadowed:
            try:
                locals = dict(locals)
                for _name in _shadowed_builtins:
                    if _name in locals and isinstance(locals[_name], (types.FunctionType, types.MethodType)):
                        del locals[_name]
            except Exception:
                pass
                    
    return _original_eval(code, globals, locals)

builtins.eval = _patched_eval

import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv

from HR.app import render_hr_agent
from TRACKX.app import render_trackx_agent
from Customer.app import render_customer_agent
from Finance.app import render_finance_agent
from orchestration.orchestrator import Orchestrator

def gather_agent_data():
    st.subheader("Enterprise Data Overview")
    st.write("Centralized real-time overview of active databases across organizational pillars.")
    
    col1, col2 = st.columns(2)
    
    # HR Data
    with col1:
        st.markdown("#### Human Resources")
        hr_path = "resume_dataset_2.csv"
        if os.path.exists(hr_path):
            try:
                hr_df = pd.read_csv(hr_path)
                st.metric("Total Resumes in Database", f"{len(hr_df):,}")
                st.dataframe(hr_df[['Name', 'Job_Role', 'Skills', 'University']].head(3), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Could not load HR data: {e}")
        else:
            st.info("No HR data available.")

    # Finance Data
    with col2:
        st.markdown("#### Finance")
        fin_path = "corporate_financial_analytics_data.csv"
        if os.path.exists(fin_path):
            try:
                fin_df = pd.read_csv(fin_path)
                st.metric("Financial Records in Database", f"{len(fin_df):,}")
                st.dataframe(fin_df[['Transaction_ID', 'Date', 'Department', 'Expense', 'Budget']].head(3), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Could not load Finance data: {e}")
        else:
            st.info("No Finance data available.")

    col3, col4 = st.columns(2)

    # Customer Data
    with col3:
        st.markdown("#### Customer Experience")
        cust_path = "customer_support_tickets_120.csv"
        if os.path.exists(cust_path):
            try:
                cust_df = pd.read_csv(cust_path)
                st.metric("Logged Support Tickets", f"{len(cust_df):,}")
                st.dataframe(cust_df[['Ticket ID', 'Customer Name', 'Product Purchased', 'Ticket Subject', 'Ticket Status']].head(3), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Could not load Customer data: {e}")
        else:
            st.info("No Customer data available.")

    # TrackX Data
    with col4:
        st.markdown("#### Performance Integrations (TrackX)")
        trackx_path = "Extended_Employee_Performance_and_Productivity_Data.csv"
        if os.path.exists(trackx_path):
            try:
                tx_df = pd.read_csv(trackx_path)
                st.metric("Employee Performance Profiles", f"{len(tx_df):,}")
                st.dataframe(tx_df[['Employee_ID', 'Department', 'Job_Title', 'Performance_Score', 'Employee_Satisfaction_Score']].head(3), width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Could not load TrackX data: {e}")
        else:
            st.info("Performance data not found.")

def render_summary():
    st.markdown("""
    <div style="padding: 3rem; background: #1e293b; color: #f8fafc; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3b82f6;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 600; letter-spacing: -0.025em; color: #f8fafc;">Enterprise AI Orchestration Platform</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; font-weight: 400; margin: 0;">Unified Intelligence for Organizational Pillars</p>
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
        st.info("#### Human Resources\nResume parsing, talent pool explorer, and simulated screening chats.")
    with col2:
        st.success("#### Performance (TrackX)\n100k employee profiles, burnout correlation charts, and standards RAG.")
    with col3:
        st.warning("#### Customer Experience\nSupport tickets, priority pies, donut resolution charts, and RAG chats.")
    with col4:
        st.error("#### Financial Analytics\nDepartmental expense distributions, transaction anomaly alerts, and forecasts.")

    st.divider()
    gather_agent_data()

def render_orchestrated_chatbot(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.2);">
        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.025em; color: white;">Centralized Enterprise AI Orchestrator</h1>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.95; font-weight: 400; color: #e0e7ff;">L1 Intent Router & L2 Specialized RAG Agent Gateway</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("Please configure your Gemini API Key in the left sidebar to use the central orchestrator.")
        return
        
    if "orch_history" not in st.session_state:
        st.session_state.orch_history = []
        
    if "current_trace" not in st.session_state:
        st.session_state.current_trace = []

    chat_col, trace_col = st.columns([7, 5], gap="large")
    
    with chat_col:
        st.subheader("Conversation Portal")
        
        # Display past messages
        for idx, msg in enumerate(st.session_state.orch_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("role") == "assistant" and msg.get("details"):
                    with st.expander("Orchestration Metadata", expanded=False):
                        st.markdown(f"**Specialist Agent Called:** `{msg.get('agent')}`")
                        st.markdown(f"**Confidence Score:** `{msg.get('confidence')}`")
                        st.markdown(f"**Tools Triggered:** `{', '.join(msg.get('tools', []))}`")
                        st.markdown(f"**Data Sources:** `{', '.join(msg.get('sources', []))}`")
                            
        # Chat input
        if prompt := st.chat_input("Ask about Finance budgets, HR Resumes, Support tickets, or performance metrics..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.orch_history.append({"role": "user", "content": prompt})
            
            # Setup tracer placeholder
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                status_placeholder = st.status("Initializing L1 Routing Engine...", expanded=True)
                
                trace_list = []
                orchestrator = Orchestrator(api_key)
                
                # Execute orchestration generator
                for event in orchestrator.orchestrate(prompt):
                    step = event.get("step")
                    msg = event.get("message")
                    details = event.get("details", "")
                    
                    trace_list.append(event)
                    st.session_state.current_trace = trace_list.copy()
                    
                    # Update progress UI dynamically
                    if step == "L1_ROUTING":
                        status_placeholder.write(f"🔍 {msg}")
                    elif step == "L1_DECISION":
                        status_placeholder.write(f"🎯 {msg}")
                        if details:
                            status_placeholder.markdown(details)
                    elif step == "L2_RAG":
                        status_placeholder.write(f"⚡ {msg}")
                    elif step == "L2_RAG_SUCCESS":
                        status_placeholder.write(f"📚 {msg}")
                        if details:
                            with status_placeholder.expander("View Retrieved Semantic Context"):
                                st.code(details, language="text")
                    elif step == "L2_RAG_EMPTY" or step == "L2_RAG_SKIP":
                        status_placeholder.write(f"⚠️ {msg}")
                    elif step == "L2_EXECUTION":
                        status_placeholder.write(f"🤖 {msg}")
                    elif step == "COMPLETED":
                        status_placeholder.update(label="✅ Request Orchestrated Successfully", state="complete", expanded=False)
                        answer = event.get("answer")
                        answer_placeholder.markdown(answer)
                        
                        # Add assistant response to history
                        st.session_state.orch_history.append({
                            "role": "assistant",
                            "content": answer,
                            "agent": event.get("routed_agent"),
                            "confidence": event.get("confidence"),
                            "tools": event.get("tools_used"),
                            "sources": event.get("sources"),
                            "details": True
                        })
                    elif step == "FAILED":
                        status_placeholder.update(label="❌ Orchestration Encountered Error", state="error", expanded=True)
                        answer = event.get("answer")
                        answer_placeholder.markdown(answer)
                        
                        st.session_state.orch_history.append({
                            "role": "assistant",
                            "content": answer,
                            "agent": event.get("routed_agent"),
                            "confidence": "Low",
                            "tools": [],
                            "sources": [],
                            "details": False
                        })
            
            st.rerun()

    with trace_col:
        st.subheader("Orchestration Call Stack & Logs")
        
        if not st.session_state.current_trace:
            st.info("Initiate a conversation to view real-time L1 router logic and L2 specialist agent traces.")
        else:
            st.markdown("""
            <style>
            .trace-card {
                background-color: #1e1b4b;
                border-left: 5px solid #818cf8;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .trace-title {
                color: #c084fc;
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 4px;
            }
            .trace-body {
                color: #cbd5e1;
                font-size: 0.85rem;
                line-height: 1.4;
            }
            </style>
            """, unsafe_allow_html=True)
            
            for ev in st.session_state.current_trace:
                step = ev.get("step")
                msg = ev.get("message")
                
                step_title = step.replace("_", " ").title()
                emoji = "⚙️"
                if "ROUTING" in step:
                    emoji = "🔍"
                elif "DECISION" in step:
                    emoji = "🎯"
                elif "RAG" in step:
                    emoji = "📚"
                elif "EXECUTION" in step:
                    emoji = "🤖"
                elif "COMPLETED" in step:
                    emoji = "✅"
                elif "FAILED" in step:
                    emoji = "❌"
                
                st.markdown(f"""
                <div class="trace-card">
                    <div class="trace-title">{emoji} {step_title}</div>
                    <div class="trace-body">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
                
        if st.button("Clear Conversation History", key="orch_reset_btn"):
            st.session_state.orch_history = []
            st.session_state.current_trace = []
            st.rerun()

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
    
    # 6 Tabs
    t1, t_orch, t2, t3, t4, t5 = st.tabs([
        "Executive Summary", 
        "Enterprise Chatbot", 
        "Human Resources", 
        "Performance Integrations", 
        "Customer Experience", 
        "Financial Analytics"
    ])
    
    with t1:
        render_summary()
    with t_orch:
        render_orchestrated_chatbot(api_key=global_api_key)
    with t2:
        render_hr_agent(api_key=global_api_key)
    with t3:
        render_trackx_agent(api_key=global_api_key)
    with t4:
        render_customer_agent(api_key=global_api_key)
    with t5:
        render_finance_agent(api_key=global_api_key)

if __name__ == "__main__":
    import streamlit.runtime as st_runtime
    if st_runtime.exists():
        main()
    else:
        import sys
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
