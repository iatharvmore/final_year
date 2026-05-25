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
        if agent in ["HR", "Finance", "Customer"]:
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
            yield {"step": "L2_RAG_SKIP", "message": "TrackX / General queries utilize integrated synthetic performance tables."}

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
                hrms_df, erp_df, crm_df, merged_df = generate_synthetic_data()
                answer = run_agent_orchestration(self.api_key, merged_df.to_string(), query_context=query)
                sources = ["Synthetic HRMS", "Synthetic ERP", "Synthetic CRM"]
                tools = ["trackx_performance_insights"]
                confidence_val = "High"
                
            else: # General Corporate Assistant
                general_prompt = f"""
                You are the master Enterprise AI Executive Assistant. 
                Answer the following corporate question professionally, synthetically combining organizational pillars if applicable.
                
                User Query: {query}
                """
                response = self.model.generate_content(general_prompt)
                answer = response.text
                sources = ["Internal Knowledge Base"]
                tools = ["llm_reasoning"]
                confidence_val = "Medium"
                
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
