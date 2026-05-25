import streamlit as st
import os
import tempfile
import pandas as pd
from HR.src.agent import ResumeAgent

import streamlit as st
import os
import tempfile
import pandas as pd
import plotly.express as px
import google.generativeai as genai
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
            box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
        }
        .stat-card-hr {
            text-align: center;
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-top: 5px solid #1e3c72;
            margin-bottom: 1.5rem;
        }
        .stat-value-hr {
            font-size: 2.2rem;
            font-weight: 800;
            color: #1e3c72;
        }
        .stat-label-hr {
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        .candidate-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container-hr">
        <h1 style="color: white; margin: 0;">HR Intelligence Portal</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Analyze talent pools, view database visualisations, and chat with candidate personas using Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load candidate database by default
    csv_path = "resume_dataset_2.csv"
    if os.path.exists(csv_path):
        try:
            df_candidates = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load candidate database: {e}")
            df_candidates = pd.DataFrame()
    else:
        st.warning("Candidate database 'resume_dataset_2.csv' not found. Displaying fallback.")
        df_candidates = pd.DataFrame()

    tab_explore, tab_original_ranker = st.tabs(["Talent Pool Explorer & Visuals", "Custom Resume Ranker (PDF/DOCX)"])
    
    with tab_explore:
        if not df_candidates.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="stat-card-hr">
                    <div class="stat-value-hr">{len(df_candidates):,}</div>
                    <div class="stat-label-hr">Total Profiles Loaded</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                avg_exp = df_candidates['Years_Experience'].mean() if 'Years_Experience' in df_candidates.columns else 0.0
                st.markdown(f"""
                <div class="stat-card-hr">
                    <div class="stat-value-hr">{avg_exp:.1f} Yrs</div>
                    <div class="stat-label-hr">Average Experience</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                unique_roles = df_candidates['Job_Role'].nunique() if 'Job_Role' in df_candidates.columns else 0
                st.markdown(f"""
                <div class="stat-card-hr">
                    <div class="stat-value-hr">{unique_roles}</div>
                    <div class="stat-label-hr">Distinct Job Roles</div>
                </div>
                """, unsafe_allow_html=True)
                
            c_left, c_right = st.columns(2)
            
            with c_left:
                st.subheader("Job Roles Distribution")
                if 'Job_Role' in df_candidates.columns:
                    role_counts = df_candidates['Job_Role'].value_counts().reset_index()
                    role_counts.columns = ['Job_Role', 'Count']
                    fig_pie = px.pie(
                        role_counts, 
                        names='Job_Role', 
                        values='Count',
                        color_discrete_sequence=px.colors.qualitative.Prism,
                        hole=0.4
                    )
                    fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                    st.plotly_chart(fig_pie, width="stretch")
            
            with c_right:
                st.subheader("Top 10 Candidate Skills")
                if 'Skills' in df_candidates.columns:
                    # Parse comma-separated skills
                    all_skills = df_candidates['Skills'].dropna().str.split(',').explode().str.strip()
                    skill_counts = all_skills.value_counts().head(10).reset_index()
                    skill_counts.columns = ['Skill', 'Count']
                    
                    fig_bar = px.bar(
                        skill_counts,
                        x='Count',
                        y='Skill',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='Blues',
                        labels={'Skill': 'Candidate Skill', 'Count': 'Count'}
                    )
                    fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, width="stretch")
            
            st.divider()
            
            st.subheader("🔍 Skill & Job Role RAG Matcher")
            search_query = st.text_input(
                "Filter candidates by skill, job role, name, or university...", 
                placeholder="e.g. Data Scientist, Python, Delhi University",
                key="hr_pool_search_query"
            )
            
            # Filtering
            filtered_df = df_candidates
            if search_query:
                q = search_query.lower()
                mask = (
                    filtered_df['Name'].str.lower().str.contains(q, na=False) |
                    filtered_df['Job_Role'].str.lower().str.contains(q, na=False) |
                    filtered_df['Skills'].str.lower().str.contains(q, na=False) |
                    filtered_df['University'].str.lower().str.contains(q, na=False)
                )
                filtered_df = filtered_df[mask]
                
            st.write(f"Showing **{len(filtered_df):,}** matching candidate profiles")
            st.dataframe(
                filtered_df[['Name', 'Job_Role', 'Years_Experience', 'Skills', 'University', 'Graduation_Year', 'Email', 'Phone']].head(50), 
                width="stretch",
                hide_index=True
            )
            
            st.divider()
            
            st.subheader("💬 Interactive Candidate AI Simulator (Short Chat)")
            st.write("Simulate a live initial chat screen or Q&A session with any candidate listed above.")
            
            candidates_to_select = filtered_df['Name'].head(30).tolist()
            if candidates_to_select:
                selected_cand = st.selectbox(
                    "Choose candidate to start screen chat:", 
                    options=candidates_to_select, 
                    key="hr_cand_interview_select"
                )
                
                # Fetch row
                cand_row = df_candidates[df_candidates['Name'] == selected_cand].iloc[0]
                
                c_details_1, c_details_2 = st.columns(2)
                with c_details_1:
                    st.markdown(f"""
                    **Selected Profile Details:**
                    * **Name:** {cand_row['Name']}
                    * **Role:** {cand_row['Job_Role']}
                    * **Experience:** {cand_row['Years_Experience']} Years
                    """)
                with c_details_2:
                    st.markdown(f"""
                    * **Education:** {cand_row['University']} (Class of {cand_row['Graduation_Year']})
                    * **Skills:** {cand_row['Skills']}
                    """)
                
                # Chat interface for selected candidate
                if "hr_chats" not in st.session_state:
                    st.session_state.hr_chats = {}
                    
                cand_chat_key = f"hr_chat_{selected_cand}"
                if cand_chat_key not in st.session_state.hr_chats:
                    st.session_state.hr_chats[cand_chat_key] = [
                        {"role": "assistant", "content": f"Hello! This is {selected_cand}. Thanks for considering my profile for the job. Feel free to ask me anything about my experience, skills, or projects!"}
                    ]
                    
                for msg in st.session_state.hr_chats[cand_chat_key]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        
                if hr_user_msg := st.chat_input("Ask candidate a screening question...", key=f"hr_user_input_{selected_cand}"):
                    st.chat_message("user").markdown(hr_user_msg)
                    st.session_state.hr_chats[cand_chat_key].append({"role": "user", "content": hr_user_msg})
                    
                    with st.spinner(f"Simulating {selected_cand}'s reply..."):
                        if not api_key:
                            st.error("Please configure the Gemini API Key.")
                        else:
                            try:
                                genai.configure(api_key=api_key)
                                model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
                                
                                prompt = f"""
                                You are simulating the selected candidate {cand_row['Name']} in a professional job screening chat.
                                Candidate Profile:
                                - Job Role: {cand_row['Job_Role']}
                                - Experience: {cand_row['Years_Experience']} Years
                                - Education: {cand_row['University']} (Graduation: {cand_row['Graduation_Year']})
                                - Key Skills: {cand_row['Skills']}
                                - Resume Text Summary: {str(cand_row.get('Resume_Text'))[:1500]}
                                
                                Respond to the recruiter's question professionally, matching the candidate's skills, experience, and background. Keep the response concise, realistic, and positive.
                                Recruiter's question:
                                "{hr_user_msg}"
                                """
                                response = model.generate_content(prompt)
                                reply = response.text
                                
                                with st.chat_message("assistant"):
                                    st.markdown(reply)
                                    
                                st.session_state.hr_chats[cand_chat_key].append({"role": "assistant", "content": reply})
                            except Exception as e:
                                st.error(f"Error generating AI screening: {e}")
                    st.rerun()
            else:
                st.info("No candidates match your filters. Adjust the filters to find and chat with candidate personas.")
        else:
            st.info("Candidate dataset is empty or could not be loaded. Please ensure resume_dataset_2.csv is present.")
            
    with tab_original_ranker:
        model_choice = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("Job Description")
            jd_text = st.text_area("Paste the job description here...", height=300, key="hr_jd_text")

        with col2:
            st.subheader("Upload Resumes")
            uploaded_files = st.file_uploader(
                "Upload PDF or DOCX resumes",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="hr_uploaded_files"
            )
            
            if uploaded_files:
                st.success(f"{len(uploaded_files)} resumes uploaded successfully!")

        if st.button("Analyze and Rank Resumes", key="hr_analyze_btn"):
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
            st.subheader("Analysis Results")
            
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
            
            tab1, tab2 = st.tabs(["Rankings", "Detailed Extraction"])
            
            with tab1:
                display_df = df[['Rank', 'File', 'Cosine Similarity']].copy()
                if not display_df.empty:
                    display_df['Cosine Similarity'] = display_df['Cosine Similarity'].map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                st.dataframe(display_df, width="stretch")
                
                output_file = "resume_analysis.xlsx"
                df.to_excel(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button(
                        label="Download Results as Excel",
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
                            st.markdown("**Education**")
                            st.info(row.get('Education', 'N/A'))
                        with c2:
                            st.markdown("**Experience**")
                            st.info(row.get('Experience', 'N/A'))
                        st.markdown("**Projects**")
                        st.info(row.get('Projects', 'N/A'))
