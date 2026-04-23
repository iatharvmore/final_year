import streamlit as st
import os
import tempfile
import pandas as pd
from HR.src.agent import ResumeAgent

def render_hr_agent(api_key=""):
    # Custom CSS
    st.markdown("""
    <style>
        .header-container-hr {
            padding: 2rem;
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .stat-card-hr {
            text-align: center;
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-value-hr {
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label-hr {
            font-size: 0.8rem;
            color: #6c757d;
            text-transform: uppercase;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container-hr">
        <h1>🚀 HR Agent: Resume Matcher</h1>
        <p>Rank and analyze resumes against your job description using Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)
    
    model_choice = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Job Description")
        jd_text = st.text_area("Paste the job description here...", height=300, key="hr_jd_text")

    with col2:
        st.subheader("📁 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="hr_uploaded_files"
        )
        
        if uploaded_files:
            st.success(f"{len(uploaded_files)} resumes uploaded successfully!")

    if st.button("🔍 Analyze and Rank Resumes", key="hr_analyze_btn"):
        if not api_key:
            st.error("Please provide a Gemini API Key in the global configuration.")
        elif not jd_text:
            st.error("Please provide a job description.")
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
        else:
            with st.spinner("Analyzing resumes... this may take a moment."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    resume_paths = []
                    for uploaded_file in uploaded_files:
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        resume_paths.append(temp_path)
                    
                    agent = ResumeAgent(
                        job_description=jd_text,
                        llm_model=model_choice,
                        output_file="analysis_results.xlsx"
                    )
                    
                    if agent.initialize():
                        results = agent.analyze_resumes(resume_paths)
                        if results:
                            st.session_state['hr_results'] = results
                            st.success("Analysis complete!")
                        else:
                            st.error("No results generated. Please check your resumes.")
                    else:
                        st.error("Failed to initialize LLM. Please check your API key.")

    if 'hr_results' in st.session_state:
        results = st.session_state['hr_results']
        valid_results = [r for r in results if r.get('Rank', -1) != -1]
        
        st.divider()
        st.subheader("📊 Analysis Results")
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"""<div class="stat-card-hr"><div class="stat-value-hr">{len(results)}</div><div class="stat-label-hr">Total Resumes</div></div>""", unsafe_allow_html=True)
        with s2:
            top_score = max([r.get('Cosine Similarity', 0) for r in valid_results]) if valid_results else 0
            st.markdown(f"""<div class="stat-card-hr"><div class="stat-value-hr">{top_score:.2f}</div><div class="stat-label-hr">Top Match Score</div></div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="stat-card-hr"><div class="stat-value-hr">{model_choice}</div><div class="stat-label-hr">Model Used</div></div>""", unsafe_allow_html=True)
        
        st.write("")
        df = pd.DataFrame(results)
        
        tab1, tab2 = st.tabs(["🏆 Rankings", "📝 Detailed Extraction"])
        
        with tab1:
            display_df = df[['Rank', 'File', 'Cosine Similarity']].copy()
            if not display_df.empty:
                display_df['Cosine Similarity'] = display_df['Cosine Similarity'].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_df, use_container_width=True)
            
            output_file = "resume_analysis.xlsx"
            df.to_excel(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 Download Results as Excel",
                    data=f,
                    file_name="HR_Resume_Analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="hr_download_btn"
                )
                
        with tab2:
            for idx, row in df.iterrows():
                with st.expander(f"Resumé: {row['File']} (Rank: {row['Rank']})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**🎓 Education**")
                        st.info(row.get('Education', 'N/A'))
                    with c2:
                        st.markdown("**💼 Experience**")
                        st.info(row.get('Experience', 'N/A'))
                    st.markdown("**🏗️ Projects**")
                    st.info(row.get('Projects', 'N/A'))
