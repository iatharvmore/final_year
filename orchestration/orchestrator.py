import os
import sys
import pandas as pd
import json
from google.generativeai import GenerativeModel
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from orchestration.rag_service import query_rag_database, initialize_rag_databases

# Set up paths so we can import agent modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Finance.modules.llm import chat_with_data, generate_summary
from Finance.modules.variance import variance
from Finance.modules.anomaly import detect_anomaly
from TRACKX.app import run_agent_orchestration
from TRACKX.data import generate_synthetic_data

def get_global_enterprise_summary():
    summary_parts = []
    
    # 1. HR
    hr_path = "resume_dataset_2.csv"
    if os.path.exists(hr_path):
        try:
            df = pd.read_csv(hr_path)
            role_str = ""
            if 'Job_Role' in df.columns:
                roles = [str(r) for r in df['Job_Role'].dropna().unique()]
                role_str = f" Job roles represented: {', '.join(roles)}."
            summary_parts.append(f"- **HR (resume_dataset_2.csv)**: Total candidate resumes in database: {len(df)}.{role_str}")
        except Exception as e:
            summary_parts.append(f"- **HR**: Error parsing database: {e}")
    else:
        summary_parts.append("- **HR**: Database file not found.")
        
    # 2. TrackX
    trackx_path = "Extended_Employee_Performance_and_Productivity_Data.csv"
    if os.path.exists(trackx_path):
        try:
            df = pd.read_csv(trackx_path)
            avg_perf = df['Performance_Score'].mean() if 'Performance_Score' in df.columns else 0.0
            avg_sat = df['Employee_Satisfaction_Score'].mean() if 'Employee_Satisfaction_Score' in df.columns else 0.0
            avg_hours = df['Work_Hours_Per_Week'].mean() if 'Work_Hours_Per_Week' in df.columns else 0.0
            summary_parts.append(
                f"- **TrackX / Employee Performance (Extended_Employee_Performance_and_Productivity_Data.csv)**:\n"
                f"  - Total Registered Employees: {len(df):,}\n"
                f"  - Average Performance Rating: {avg_perf:.2f} / 5.0\n"
                f"  - Average Employee Satisfaction Score: {avg_sat:.2f} / 5.0\n"
                f"  - Average Work Hours per Week: {avg_hours:.2f} hours"
            )
        except Exception as e:
            summary_parts.append(f"- **TrackX**: Error parsing database: {e}")
    else:
        summary_parts.append("- **TrackX**: Database file not found.")

    # 3. Customer
    cust_path = "customer_support_tickets_120.csv" if os.path.exists("customer_support_tickets_120.csv") else "customer_support_tickets.csv"
    if os.path.exists(cust_path):
        try:
            df = pd.read_csv(cust_path)
            total_tickets = len(df)
            status_str = ""
            df.columns = df.columns.str.strip()
            status_col = [col for col in df.columns if 'status' in col.lower()]
            if status_col:
                status_counts = df[status_col[0]].value_counts().to_dict()
                status_str = f" Ticket statuses: {status_counts}."
            summary_parts.append(f"- **Customer Experience ({cust_path})**: Total support tickets logged: {total_tickets:,}.{status_str}")
        except Exception as e:
            summary_parts.append(f"- **Customer Experience**: Error parsing database: {e}")
    else:
        summary_parts.append("- **Customer Experience**: Database file not found.")

    # 4. Finance
    fin_path = "corporate_financial_analytics_data.csv"
    if os.path.exists(fin_path):
        try:
            df = pd.read_csv(fin_path)
            df.columns = df.columns.str.strip()
            df.columns = df.columns.str.lower()
            total_exp = df['expense'].sum() if 'expense' in df.columns else 0.0
            total_bud = df['budget'].sum() if 'budget' in df.columns else 0.0
            total_var = total_exp - total_bud
            summary_parts.append(
                f"- **Financial Analytics (corporate_financial_analytics_data.csv)**:\n"
                f"  - Total Transactions: {len(df):,}\n"
                f"  - Total Expense: ${total_exp:,.2f}\n"
                f"  - Total Budget: ${total_bud:,.2f}\n"
                f"  - Net Variance: ${total_var:,.2f} (Expenses under/over budget)"
            )
        except Exception as e:
            summary_parts.append(f"- **Financial Analytics**: Error parsing database: {e}")
    else:
        summary_parts.append("- **Financial Analytics**: Database file not found.")

    return "\n\n".join(summary_parts)

class Orchestrator:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = GenerativeModel(self.model_name)
        
        # Make sure RAG database is initialized
        initialize_rag_databases()

    def orchestrate(self, query, session_id="default_session"):
        """
        Orchestrates the L1 routing and L2 agent invocation.
        Yields trace dictionaries to allow Streamlit to update the UI in real-time.
        """
        yield {"step": "L1_ROUTING", "message": "Analyzing query and intent in L1 Orchestrator..."}
        
        # 1. Classify the user query
        routing_prompt = f"""
        You are the L1 Orchestrator of an Enterprise AI Platform. Your job is to classify the user's query and route it to the appropriate L2 departmental agent.
        
        The L2 agents and their domains are:
        1. **HR**: Candidate resumes, hiring, applicant skills, phone numbers, emails, university analysis (uses resume_dataset_2.csv).
        2. **Finance**: Budget variance, expense tracking, invoices, region transactions, vendor analysis, financial forecasts (uses corporate_financial_analytics_data.csv).
        3. **Customer**: Customer support tickets, feedback sentiment, support channel queries (uses customer_support_tickets_120.csv).
        4. **TrackX**: Integrated employee performance tracking across HRMS, ERP, and CRM (attendance, billable hours, sales deal closure, client satisfaction).
        5. **General**: Generic greetings, general corporate questions, or queries spanning multiple departments.
        
        Analyze the query: "{query}"
        
        Respond ONLY with a valid JSON block containing:
        - "agent": The selected agent string ("HR", "Finance", "Customer", "TrackX", or "General")
        - "confidence": A confidence score between 0.0 and 1.0
        - "reasoning": A brief explanation of why this agent was selected.
        
        Format your response as a raw JSON string block without markdown formatting or code blocks.
        """
        try:
            response = self.model.generate_content(routing_prompt)
            # Remove any possible ```json wrapping if returned
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            routing_decision = json.loads(clean_text)
        except Exception as e:
            print(f"L1 Routing classification failed: {e}")
            routing_decision = {"agent": "General", "confidence": 0.5, "reasoning": "Fallback due to classification error."}
            
        agent = routing_decision.get("agent", "General")
        confidence = routing_decision.get("confidence", 1.0)
        reasoning = routing_decision.get("reasoning", "")
        
        yield {
            "step": "L1_DECISION", 
            "message": f"L1 Router Decision: Routed to **{agent} Agent** (Confidence: {confidence:.2f}).",
            "details": f"**Reasoning**: {reasoning}"
        }
        
        # 2. Retrieve context via RAG
        yield {"step": "L2_RAG", "message": f"L2 Agent: Querying Central FAISS database for **{agent}** context..."}
        
        rag_context = ""
        retrieved_docs = []
        if agent in ["HR", "Finance", "Customer", "TrackX"]:
            retrieved_docs = query_rag_database(agent, query, k=4)
            if retrieved_docs:
                rag_context = "\n".join([f"- {doc.page_content}" for doc in retrieved_docs])
                yield {
                    "step": "L2_RAG_SUCCESS", 
                    "message": f"Retrieved {len(retrieved_docs)} semantic context chunks from central vector store.",
                    "details": rag_context
                }
            else:
                yield {"step": "L2_RAG_EMPTY", "message": "No matching semantic context found in central vector store. Using general knowledge."}
        else:
            yield {"step": "L2_RAG_SKIP", "message": "General queries utilize system-wide database statistics."}

        # 3. Invoke L2 Agent execution
        yield {"step": "L2_EXECUTION", "message": f"L2 Agent: Executing **{agent}** intelligence loop..."}
        
        try:
            if agent == "Finance":
                # Finance RAG & data loading
                csv_path = "corporate_financial_analytics_data.csv"
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                else:
                    df = pd.read_csv("Finance/data/sample_finance.csv")
                
                df.columns = df.columns.str.lower().str.strip()
                var_df = variance(df.copy())
                an_df = detect_anomaly(var_df.copy())
                summary_data = generate_summary(an_df)
                
                if rag_context:
                    summary_data += f"\n\nAdditional Transaction Context (FAISS):\n{rag_context}"
                
                # Execute finance agent chat
                resp = chat_with_data(query, summary_data, session_id, df)
                answer = resp.answer
                sources = list(set([doc.metadata.get("source", "corporate_financial_analytics_data.csv") for doc in retrieved_docs]))
                tools = resp.tools_used or ["retrieval_tool"]
                confidence_val = resp.confidence or "High"
                
            elif agent == "HR":
                # HR Candidate Retrieval
                # Build context-rich answer using Gemini and RAG from resume_dataset_2.csv
                hr_prompt = f"""
                You are an expert HR Specialist and Technical Recruiter.
                Use the following retrieved candidate resume details to precisely answer the user's recruitment query.
                
                Retrieved Candidates (RAG):
                {rag_context}
                
                User Query: {query}
                
                Answer professionally, referencing candidate names, skills, graduation years, and experiences.
                """
                response = self.model.generate_content(hr_prompt)
                answer = response.text
                sources = ["resume_dataset_2.csv"]
                tools = ["hr_resume_retrieval"]
                confidence_val = "High"
                
            elif agent == "Customer":
                # Customer Support analytics/query
                cust_prompt = f"""
                You are an intelligent Customer Support Specialist.
                Use the following retrieved support tickets and sentiment details to answer the user query.
                
                Retrieved Ticket Context (RAG):
                {rag_context}
                
                User Query: {query}
                
                Summarize ticket statuses, sentiments, or standard procedures concisely.
                """
                response = self.model.generate_content(cust_prompt)
                answer = response.text
                sources = ["customer_support_tickets_120.csv"]
                tools = ["customer_sentiment_retrieval"]
                confidence_val = "High"
                
            elif agent == "TrackX":
                # Holistic performance integration
                trackx_csv = "Extended_Employee_Performance_and_Productivity_Data.csv"
                if os.path.exists(trackx_csv):
                    df_perf = pd.read_csv(trackx_csv)
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
                
                if rag_context:
                    summary_text += f"\n\nRetrieved Employee Performance Records (FAISS RAG):\n{rag_context}"
                
                answer = run_agent_orchestration(self.api_key, summary_text, query_context=query)
                sources = ["Extended_Employee_Performance_and_Productivity_Data.csv"] if os.path.exists(trackx_csv) else ["Synthetic HRMS", "Synthetic ERP", "Synthetic CRM"]
                tools = ["trackx_performance_insights"]
                confidence_val = "High"
                
            else: # General Corporate Assistant
                global_summary = get_global_enterprise_summary()
                general_prompt = f"""
                You are the master Enterprise AI Executive Assistant. 
                You have access to the global enterprise database summaries below.
                
                When the user asks for figures, counts, financials, or employee statistics:
                1. You MUST provide exact numbers, counts, and metrics from the provided summaries.
                2. Do NOT say the numbers are not available or suggest consulting the departments if the numbers are present in the summary below.
                3. Be professional, direct, and factual. Answer both parts of any multi-part questions clearly.
                
                Global Enterprise Summaries:
                {global_summary}
                
                User Query: {query}
                """
                response = self.model.generate_content(general_prompt)
                answer = response.text
                sources = ["resume_dataset_2.csv", "Extended_Employee_Performance_and_Productivity_Data.csv", "customer_support_tickets_120.csv", "corporate_financial_analytics_data.csv"]
                tools = ["global_data_synthesis"]
                confidence_val = "High"
                
            yield {
                "step": "COMPLETED", 
                "message": f"Successfully generated answer.",
                "answer": answer,
                "sources": sources,
                "tools_used": tools,
                "confidence": confidence_val,
                "routed_agent": agent
            }
            
        except Exception as e:
            err_msg = f"Failed to execute L2 agent logic: {e}"
            yield {
                "step": "FAILED", 
                "message": err_msg,
                "answer": f"I ran into an issue handling that query via the {agent} agent. Details: {e}",
                "sources": [],
                "tools_used": [],
                "confidence": "Low",
                "routed_agent": agent
            }
