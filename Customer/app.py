import streamlit as st
import pandas as pd
import json
import os
import google.generativeai as genai

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

def handle_query(api_key, query, kb_data):
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    kb_context = json.dumps(kb_data["knowledge_base"], indent=2)
    
    prompt = f"""
    You are an intelligent Customer Support Agent. Use the following Knowledge Base to answer the user's query.
    
    Knowledge Base:
    {kb_context}
    
    User Query: "{query}"
    
    Instructions:
    1. If the query matches an entry in the Knowledge Base, provide the 'Standard Response'.
    2. If the query is complex or requires escalation (as per keywords/complexity), state that you are escalating to a human agent.
    3. If the query is out-of-scope, use the 'out_of_scope_response' from the knowledge base.
    4. Maintain a polite and professional tone.
    5. If clarification is needed, ask follow-up questions.
    """
    
    response = model.generate_content(prompt)
    return response.text

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
        
        # Dummy data
        dummy_data = pd.DataFrame({
            "TicketID": ["TCK-101", "TCK-102", "TCK-103", "TCK-104", "TCK-105"],
            "Customer": ["Acme Corp", "Globex", "Initech", "Umbrella Corp", "Soylent"],
            "Issue": ["Login timeout error", "Billing discrepancy", "Feature request: Dark mode", "API rate limit exceeded", "App crashes on launch"],
            "Status": ["Open", "Resolved", "Pending", "Open", "Critical"],
            "Sentiment": ["Frustrated", "Neutral", "Positive", "Angry", "Angry"]
        })
        
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
        st.subheader("Interactive Support Chatbot")
        st.info("Ask questions about billing, technical issues, or account management. The agent uses the Knowledge Base to assist you.")
        
        kb_data = load_knowledge_base()
        
        if kb_data:
            # Initialize chat history
            if "customer_messages" not in st.session_state:
                st.session_state.customer_messages = [
                    {"role": "assistant", "content": "Hello! I'm your Support Assistant. How can I help you today?"}
                ]

            # Display chat messages from history on app rerun
            for message in st.session_state.customer_messages:
                with st.chat_message(message["role"]):
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
                    prompt_suggested = "The API is returning a 504 error."
            with cols[2]:
                if st.button("Reset Password", use_container_width=True):
                    prompt_suggested = "How do I reset my password?"
            
            # React to user input
            prompt = st.chat_input("What is on your mind?")
            
            # If a suggested button was clicked, use that as the prompt
            if 'prompt_suggested' in locals() and prompt_suggested:
                prompt = prompt_suggested

            if prompt:
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.customer_messages.append({"role": "user", "content": prompt})

                with st.spinner("Support Agent is typing..."):
                    try:
                        # Prepare context from history
                        history_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.customer_messages[-5:]])
                        kb_context = json.dumps(kb_data["knowledge_base"], indent=2)
                        
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel(model_choice)
                        
                        agent_prompt = f"""
                        You are an intelligent Customer Support Chatbot. 
                        Use the following Knowledge Base to answer the user's queries accurately.
                        
                        Knowledge Base:
                        {kb_context}
                        
                        Current Conversation History:
                        {history_context}
                        
                        Instructions:
                        1. If the user query relates to an entry in the Knowledge Base, use the 'Standard Response' but adapt it to the conversation flow.
                        2. If the issue is complex or the user seems frustrated, offer to escalate to a human agent.
                        3. If the user asks something out of scope, use the 'out_of_scope_response'.
                        4. This is a multi-turn conversation. Be helpful, concise, and professional.
                        5. If the user provides a detail (like an invoice number), acknowledge it and proceed with the knowledge base logic.
                        """
                        
                        response = model.generate_content(agent_prompt)
                        full_response = response.text
                        
                        # Display assistant response in chat message container
                        with st.chat_message("assistant"):
                            st.markdown(full_response)
                        
                        # Add assistant response to chat history
                        st.session_state.customer_messages.append({"role": "assistant", "content": full_response})
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            if st.button("Clear Chat History", key="clear_customer_chat"):
                st.session_state.customer_messages = [
                    {"role": "assistant", "content": "Hello! I'm your Support Assistant. How can I help you today?"}
                ]
                st.rerun()

            with st.expander("Reference: Knowledge Base Content"):
                st.json(kb_data)
        else:
            st.error("Knowledge base file not found.")
