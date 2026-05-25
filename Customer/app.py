import streamlit as st
import pandas as pd
import json
import os
import uuid
import plotly.express as px
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
    <div style="padding: 2rem; background: linear-gradient(90deg, #f2994a 0%, #f2c94c 100%); color: white; border-radius: 10px; margin-bottom: 2rem; text-align: center; box-shadow: 0 4px 15px rgba(242, 153, 74, 0.2);">
        <h1 style="color: white; margin: 0;">Customer Agent Dashboard</h1>
        <p style="margin: 5px 0 0 0; opacity: 0.95;">AI-driven customer experience and support ticket analysis platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not api_key:
        st.warning("Please configure your Gemini API Key in the global configuration.")
        return

    model_choice = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Load customer tickets dataset by default
    csv_path = "customer_support_tickets_120.csv"
    if os.path.exists(csv_path):
        try:
            df_tickets = pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Failed to load customer tickets database: {e}")
            df_tickets = pd.DataFrame()
    else:
        st.warning("Customer support tickets dataset 'customer_support_tickets_120.csv' not found.")
        df_tickets = pd.DataFrame()

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Customer Feedback & Analytics", "Interactive Support Chatbot (RAG Agent)"])

    with tab1:
        st.subheader("Support Ticket Database Insights")
        
        if not df_tickets.empty:
            # Metrics
            total_tickets = len(df_tickets)
            avg_csat = df_tickets['Customer Satisfaction Rating'].mean() if 'Customer Satisfaction Rating' in df_tickets.columns else 0.0
            open_tickets = len(df_tickets[df_tickets['Ticket Status'].str.lower().str.contains('pending|open', na=False)]) if 'Ticket Status' in df_tickets.columns else 0
            
            s1, s2, s3 = st.columns(3)
            with s1:
                st.metric("Total Support Tickets", f"{total_tickets:,}")
            with s2:
                st.metric("Average Satisfaction Score", f"{avg_csat:.2f} / 5.0")
            with s3:
                st.metric("Pending/Open Tickets", f"{open_tickets:,}")
                
            # Plotly Charts
            col_chart_1, col_chart_2, col_chart_3 = st.columns(3)
            
            with col_chart_1:
                st.markdown("**Tickets by Product Purchased**")
                if 'Product Purchased' in df_tickets.columns:
                    prod_counts = df_tickets['Product Purchased'].value_counts().reset_index()
                    prod_counts.columns = ['Product', 'Tickets']
                    fig_prod = px.bar(
                        prod_counts,
                        x='Tickets',
                        y='Product',
                        orientation='h',
                        color='Tickets',
                        color_continuous_scale='Oranges'
                    )
                    fig_prod.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250, yaxis={'categoryorder':'total ascending'}, showlegend=False)
                    st.plotly_chart(fig_prod, width="stretch")
                    
            with col_chart_2:
                st.markdown("**Priority Distribution**")
                if 'Ticket Priority' in df_tickets.columns:
                    prio_counts = df_tickets['Ticket Priority'].value_counts().reset_index()
                    prio_counts.columns = ['Priority', 'Count']
                    fig_prio = px.pie(
                        prio_counts,
                        names='Priority',
                        values='Count',
                        color_discrete_sequence=px.colors.sequential.Oranges_r,
                        hole=0.4
                    )
                    fig_prio.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
                    st.plotly_chart(fig_prio, width="stretch")
                    
            with col_chart_3:
                st.markdown("**Ticket Status Distribution**")
                if 'Ticket Status' in df_tickets.columns:
                    status_counts = df_tickets['Ticket Status'].value_counts().reset_index()
                    status_counts.columns = ['Status', 'Count']
                    fig_status = px.pie(
                        status_counts,
                        names='Status',
                        values='Count',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig_status.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250)
                    st.plotly_chart(fig_status, width="stretch")
            
            st.divider()
            
            st.subheader("🔍 Active Ticket Explorer & Filter")
            ticket_search = st.text_input(
                "Search tickets by Product, Customer Name, Subject or Status...", 
                placeholder="e.g. Dell XPS, Critical, Refund",
                key="customer_ticket_search"
            )
            
            filtered_tickets = df_tickets
            if ticket_search:
                ts = ticket_search.lower()
                mask = (
                    filtered_tickets['Customer Name'].str.lower().str.contains(ts, na=False) |
                    filtered_tickets['Product Purchased'].str.lower().str.contains(ts, na=False) |
                    filtered_tickets['Ticket Subject'].str.lower().str.contains(ts, na=False) |
                    filtered_tickets['Ticket Status'].str.lower().str.contains(ts, na=False) |
                    filtered_tickets['Ticket Priority'].str.lower().str.contains(ts, na=False)
                )
                filtered_tickets = filtered_tickets[mask]
                
            st.write(f"Displaying **{len(filtered_tickets)}** support tickets")
            st.dataframe(
                filtered_tickets[['Ticket ID', 'Customer Name', 'Customer Email', 'Product Purchased', 'Ticket Subject', 'Ticket Priority', 'Ticket Status']].head(50),
                width="stretch",
                hide_index=True
            )
            
            st.divider()
            
            st.subheader("🤖 AI Customer Experience Analyzer")
            if st.button("Generate AI Customer Sentiment Report", key="customer_analyze_btn"):
                with st.spinner("AI Customer Agent is analyzing..."):
                    try:
                        insights = run_customer_agent(api_key, filtered_tickets.head(30).to_string())
                        st.markdown("### Customer Experience Insights")
                        st.markdown(insights)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("No tickets dataset found or loaded.")

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
            if st.button("Update Payment", width="stretch"):
                prompt_suggested = "How do I update my credit card?"
        with cols[1]:
            if st.button("API 504 Error", width="stretch"):
                prompt_suggested = "The API is returning a 504 error consistently."
        with cols[2]:
            if st.button("Check My Subscription", width="stretch"):
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
