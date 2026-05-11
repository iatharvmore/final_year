import streamlit as st
import pandas as pd
import json
import os
import uuid
from datetime import datetime
import google.generativeai as genai

from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from .rag_utils import load_chat_history, save_message, clear_chat_history, index_knowledge_base, process_documents, load_tickets
from .agent_tools import get_tools

class AgentResponse(BaseModel):
    question: str = Field(description="The user's original question.")
    answer: str = Field(description="The detailed response to the user's query.")
    sources: List[str] = Field(default_factory=list, description="List of document sources or context used.")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used.")
    confidence: str = Field(description="Confidence level of the response (e.g., High, Medium, Low).")
    timestamp: str = Field(description="The timestamp of the response in ISO format.")

def load_knowledge_base():
    path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def run_customer_agent(api_key, data):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = f"""
    You are a Customer Experience AI Agent. Review the following recent feedback and support tickets.
    Identify common pain points, categorize feedback sentiment, and propose improvements.
    
    Feedback Data:
    {data}
    
    Format response:
    1. Sentiment Overview
    2. Key Pain Points
    3. Suggested Improvements
    """
    response = model.generate_content(prompt)
    return response.text

from .rag_utils import load_chat_history, save_message, clear_chat_history, index_knowledge_base, process_documents, load_tickets, process_csv_to_sql

def process_uploaded_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(".csv"):
            success = process_csv_to_sql(uploaded_file)
            if not success:
                st.error(f"Failed to load CSV: {uploaded_file.name}")
        else:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=content, metadata={"source": uploaded_file.name}))
    if docs:
        process_documents(docs)

def render_customer_agent(api_key=""):
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(90deg, #f2994a 0%, #f2c94c 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center;">
        <h1>Customer Agent Dashboard</h1>
        <p>AI-driven customer experience and support ticket analysis platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("Please configure your Gemini API Key in the global configuration.")
        return

    model_choice = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Sentiment Analysis", "Query Handling Agent"])

    with tab1:
        st.subheader("Recent Customer Feedback")
        
        # Base dummy data
        dummy_df = pd.DataFrame({
            "TicketID": ["TCK-101", "TCK-102", "TCK-103", "TCK-104", "TCK-105"],
            "Customer": ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Soylent"],
            "Issue": ["Login timeout error", "Billing discrepancy", "Feature request: Dark mode", "API rate limit exceeded", "App crashes on launch"],
            "Status": ["Open", "Resolved", "Pending", "Open", "Critical"],
            "Sentiment": ["Frustrated", "Neutral", "Positive", "Angry", "Angry"]
        })
        
        # Load real tickets from agent tool usage
        real_tickets = load_tickets()
        if real_tickets:
            real_df = pd.DataFrame(real_tickets)
            dummy_data = pd.concat([real_df, dummy_df], ignore_index=True)
        else:
            dummy_data = dummy_df
            
        st.dataframe(dummy_data, use_container_width=True)
        st.divider()
        if st.button("Analyze Customer Sentiment", key="customer_analyze_btn"):
            with st.spinner("AI Customer Agent is analyzing..."):
                try:
                    insights = run_customer_agent(api_key, dummy_data.to_string())
                    st.markdown("### Customer Experience Insights")
                    st.markdown(insights)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        st.subheader("Interactive Support Chatbot (RAG Agent)")
        st.info("Ask questions about billing, technical issues, or account management. The agent uses tools and FAISS retrieval to assist you.")
        
        # Initialize session ID
        if "customer_session_id" not in st.session_state:
            st.session_state.customer_session_id = str(uuid.uuid4())
            # Initialize the base knowledge base on first run
            index_knowledge_base()

        # Document Uploader (Collapsible)
        with st.expander("Upload Documents for Knowledge Base"):
            uploaded_files = st.file_uploader("Upload TXT/CSV/JSON files", type=["txt", "csv", "json"], accept_multiple_files=True)
            if st.button("Process & Index Documents"):
                if uploaded_files:
                    with st.spinner("Indexing documents into FAISS..."):
                        process_uploaded_files(uploaded_files)
                    st.success("Documents successfully indexed!")
                else:
                    st.warning("Please upload files first.")

        # Load Chat History from SQLite
        session_id = st.session_state.customer_session_id
        db_messages = load_chat_history(session_id)
        
        # Display chat messages from DB
        for message in db_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    try:
                        # Attempt to parse as JSON to render beautifully
                        data = json.loads(message["content"])
                        st.markdown(data.get("answer", ""))
                        with st.expander("View Agent Metadata"):
                            st.json(data)
                    except json.JSONDecodeError:
                        st.markdown(message["content"])
                else:
                    st.markdown(message["content"])

        # Suggested Questions
        st.markdown("---")
        st.caption("Suggested Questions:")
        cols = st.columns(3)
        with cols[0]:
            if st.button("Update Payment", use_container_width=True):
                prompt_suggested = "How do I update my credit card?"
        with cols[1]:
            if st.button("API 504 Error", use_container_width=True):
                prompt_suggested = "The API is returning a 504 error consistently."
        with cols[2]:
            if st.button("Check My Subscription", use_container_width=True):
                prompt_suggested = "Can you check the subscription for john@example.com?"
        
        # React to user input
        prompt = st.chat_input("What is on your mind?")
        if 'prompt_suggested' in locals() and prompt_suggested:
            prompt = prompt_suggested

        if prompt:
            st.chat_message("user").markdown(prompt)
            save_message(session_id, "user", prompt)

            with st.spinner("Support Agent is typing..."):
                try:
                    # Setup LangChain Agent
                    llm = ChatGoogleGenerativeAI(model=model_choice, google_api_key=api_key, temperature=0.2)
                    tools = get_tools()
                    
                    system_prompt = '''
                    You are an intelligent Customer Support Chatbot with access to multiple tools.
                    Always use the appropriate tool to find answers or perform actions.
                    When answering, synthesize the information retrieved from the tools.
                    '''
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    agent = create_tool_calling_agent(llm, tools, prompt_template)
                    agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
                    
                    # Convert DB messages to LangChain messages
                    chat_history = []
                    for m in db_messages[-10:]: # last 10 messages
                        if m["role"] == "user":
                            chat_history.append(HumanMessage(content=m["content"]))
                        else:
                            try:
                                ans = json.loads(m["content"]).get("answer", m["content"])
                                chat_history.append(AIMessage(content=ans))
                            except:
                                chat_history.append(AIMessage(content=m["content"]))

                    # Execute Agent
                    response = agent_executor.invoke({"input": prompt, "chat_history": chat_history})
                    
                    # Extract Data
                    final_answer = response["output"]
                    intermediate_steps = response.get("intermediate_steps", [])
                    
                    tools_used = list(set([step[0].tool for step in intermediate_steps]))
                    sources = []
                    for step in intermediate_steps:
                        if step[0].tool == "retrieval_tool":
                            # Extremely naive extraction of sources from the string output for demo purposes
                            if "[Source:" in step[1]:
                                parts = step[1].split("[Source:")
                                for p in parts[1:]:
                                    src = p.split("]")[0].strip()
                                    if src not in sources:
                                        sources.append(src)
                    
                    # Ensure JSON structure
                    structured_output = AgentResponse(
                        question=prompt,
                        answer=final_answer,
                        sources=sources,
                        tools_used=tools_used,
                        confidence="High" if tools_used else "Medium",
                        timestamp=datetime.utcnow().isoformat()
                    )
                    
                    json_response = structured_output.model_dump_json()
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(structured_output.answer)
                        with st.expander("View Agent Metadata"):
                            st.json(json.loads(json_response))
                    
                    # Save to DB
                    save_message(session_id, "assistant", json_response)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.button("Clear Chat History", key="clear_customer_chat"):
            clear_chat_history(session_id)
            # Re-initialize with a greeting
            save_message(session_id, "assistant", json.dumps(AgentResponse(
                question="",
                answer="Hello! I'm your RAG-powered Support Assistant. How can I help you today?",
                sources=[],
                tools_used=[],
                confidence="High",
                timestamp=datetime.utcnow().isoformat()
            ).model_dump()))
            st.rerun()

        with st.expander("Reference: Knowledge Base Content"):
            kb_data = load_knowledge_base()
            if kb_data:
                st.json(kb_data)
            else:
                st.write("No default knowledge base found.")
