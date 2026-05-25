import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
from TRACKX.data import generate_synthetic_data
from TRACKX.rag_utils import process_and_index_performance_doc, retrieve_trackx_context

def run_agent_orchestration(api_key, data, query_context=""):
    genai.configure(api_key=api_key)
    import os
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)
    
    rag_info = ""
    if query_context:
        docs = retrieve_trackx_context(query_context, k=3)
        if docs:
            rag_info = "\n\nRetrieved Performance Benchmarks / Standards Context:\n" + "\n".join(
                [f"- {d.page_content} (Source: {d.metadata.get('source')})" for d in docs]
            )
            
    prompt = f"""
    You are TrackX, an enterprise performance intelligence AI agent. 
    Analyze the following holistic performance data spanning HRMS, ERP, and CRM domains. 
    
    Identify key insights, flag anomalies (e.g., low attendance, budget overruns, unmet targets), 
    and provide actionable recommendations tailored to each employee or department.{rag_info}
    
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
    <div style="padding: 2rem; background: linear-gradient(90deg, #1fa2ff 0%, #12d8fa 50%, #a6ffcb 100%); color: #000; border-radius: 10px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(31, 162, 255, 0.2);">
        <h1 style="color: black; margin: 0;">TrackX Agent Dashboard</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9; color: black; font-weight: 500;">Monitor employee performance, detect burnout, and analyze productivity metrics across 100,000 profiles.</p>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.warning("Please configure your Gemini API Key in the left sidebar to use TrackX.")
        return

    import os
    csv_path = "Extended_Employee_Performance_and_Productivity_Data.csv"
    if os.path.exists(csv_path):
        try:
            df_perf = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load performance dataset: {e}")
            df_perf = pd.DataFrame()
    else:
        st.warning("Employee Performance database 'Extended_Employee_Performance_and_Productivity_Data.csv' not found. Displaying fallback synthetic data.")
        df_perf = pd.DataFrame()

    tab_analytics, tab_grid, tab_agent = st.tabs(["Analytics & Burnout Plots", "Talent Database Grid View", "Performance Intelligence Agent (RAG)"])

    with tab_analytics:
        if not df_perf.empty:
            # Metrics
            total_emp = len(df_perf)
            avg_perf = df_perf['Performance_Score'].mean() if 'Performance_Score' in df_perf.columns else 0.0
            avg_sat = df_perf['Employee_Satisfaction_Score'].mean() if 'Employee_Satisfaction_Score' in df_perf.columns else 0.0
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Total Registered Employees", f"{total_emp:,}")
            with m2:
                st.metric("Average Performance Rating", f"{avg_perf:.2f} / 5.0")
            with m3:
                st.metric("Average Employee Satisfaction", f"{avg_sat:.2f} / 5.0")
                
            # Plotly Charts
            col_chart_1, col_chart_2 = st.columns(2)
            
            with col_chart_1:
                st.subheader("Performance Scores Distribution")
                if 'Performance_Score' in df_perf.columns:
                    perf_counts = df_perf['Performance_Score'].value_counts().reset_index()
                    perf_counts.columns = ['Score', 'Count']
                    fig_perf = px.bar(
                        perf_counts,
                        x='Score',
                        y='Count',
                        color='Count',
                        color_continuous_scale='Mint',
                        labels={'Score': 'Performance Score', 'Count': 'Number of Employees'}
                    )
                    fig_perf.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_perf, width="stretch")
                    
            with col_chart_2:
                st.subheader("Employee Burnout Correlation")
                if 'Work_Hours_Per_Week' in df_perf.columns and 'Employee_Satisfaction_Score' in df_perf.columns:
                    # Sample 1,000 rows to prevent browser lagging while keeping high accuracy
                    df_sample = df_perf.sample(n=min(1000, len(df_perf)), random_state=42)
                    fig_scatter = px.scatter(
                        df_sample,
                        x='Work_Hours_Per_Week',
                        y='Employee_Satisfaction_Score',
                        color='Performance_Score',
                        color_continuous_scale='Viridis',
                        labels={'Work_Hours_Per_Week': 'Work Hours / Week', 'Employee_Satisfaction_Score': 'Satisfaction Score'}
                    )
                    fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_scatter, width="stretch")
            
            st.subheader("Gender Representation Across Departments")
            if 'Gender' in df_perf.columns and 'Department' in df_perf.columns:
                gender_dept = df_perf.groupby(['Department', 'Gender']).size().reset_index(name='Count')
                fig_gender = px.bar(
                    gender_dept,
                    x='Department',
                    y='Count',
                    color='Gender',
                    barmode='group',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_gender.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_gender, width="stretch")
        else:
            st.info("No performance analytics available.")

    with tab_grid:
        if not df_perf.empty:
            st.info("Showing the first 100 rows out of 100,000 registered employees in the TrackX database.")
            st.dataframe(df_perf.head(100), width="stretch", hide_index=True)
        else:
            # Fallback synthetic data
            hrms_df, erp_df, crm_df, merged_df = generate_synthetic_data()
            st.subheader("Enterprise Data Integrations (Synthetic Fallback)")
            t_int, t_hrms, t_erp, t_crm = st.tabs(["Integrated View", "HRMS Data", "ERP Data", "CRM Data"])
            with t_int:
                st.dataframe(merged_df, width="stretch")
            with t_hrms:
                st.dataframe(hrms_df, width="stretch")
            with t_erp:
                st.dataframe(erp_df, width="stretch")
            with t_crm:
                st.dataframe(crm_df, width="stretch")

    with tab_agent:
        st.subheader("Performance Intelligence Agent (RAG Enabled)")

        # RAG Upload Expandable Panel
        with st.expander("Upload Performance Guidelines, Objectives, or Benchmarks (RAG)"):
            uploaded_files = st.file_uploader(
                "Upload standards (PDF, DOCX, TXT) to index into TrackX Knowledge Base", 
                type=["pdf", "docx", "txt"], 
                accept_multiple_files=True,
                key="trackx_uploader"
            )
            if st.button("Process & Index Standards", key="trackx_index_btn"):
                if uploaded_files:
                    with st.spinner("Indexing standard benchmarks into FAISS vector store..."):
                        total_chunks = 0
                        for f in uploaded_files:
                            total_chunks += process_and_index_performance_doc(f)
                    st.success(f"Indexed successfully! {total_chunks} chunk(s) added to TrackX FAISS index.")
                else:
                    st.warning("Please upload a file first.")

        st.write("")
        query_context = st.text_input(
            "Standard Guidelines Context Query (Optional)", 
            placeholder="e.g. Sales KPIs, QA expectations, attendance regulations...",
            key="trackx_query_context"
        )

        if st.button("Generate Insights & Recommendations", type="primary", key="trackx_generate_btn"):
            with st.spinner("TrackX Agents are analyzing performance metrics using FAISS standards RAG..."):
                try:
                    if not df_perf.empty:
                        # Build a highly informative, lightweight aggregate summary of the 100,000 employees
                        summary_text = f"""
                        Holistic Performance Dataset Stats:
                        - Total Employees: {len(df_perf):,}
                        - Avg Performance Rating: {df_perf['Performance_Score'].mean():.2f} / 5.0
                        - Avg Employee Satisfaction Score: {df_perf['Employee_Satisfaction_Score'].mean():.2f} / 5.0
                        - Avg Work Hours / Week: {df_perf['Work_Hours_Per_Week'].mean():.2f} hours
                        - Avg Monthly Salary: ${df_perf['Monthly_Salary'].mean():.2f}
                        - Total Promotions Given: {df_perf['Promotions'].sum():,}
                        - Total Sick Days Taken: {df_perf['Sick_Days'].sum():,}
                        - Average Years at Company: {df_perf['Years_At_Company'].mean():.2f} years
                        - Department Performance Score Average:
                        {df_perf.groupby('Department')['Performance_Score'].mean().to_string()}
                        - Department Satisfaction Average:
                        {df_perf.groupby('Department')['Employee_Satisfaction_Score'].mean().to_string()}
                        """
                    else:
                        hrms_df, erp_df, crm_df, merged_df = generate_synthetic_data()
                        summary_text = merged_df.to_string()
                        
                    insights = run_agent_orchestration(api_key, summary_text, query_context=query_context)
                    st.markdown("### Agent Insights")
                    st.markdown(insights)
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")
