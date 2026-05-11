import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from Finance.modules.tools import calculator_tool, retrieval_tool, finance_utility_tool
from Finance.modules.schemas import FinanceAgentResponse
from Finance.modules.memory import save_message, load_history, log_retrieval

def get_agent():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # We use a standard Chat model
    llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0.2)
    
    # LangGraph ReAct agent with tools
    tools = [calculator_tool, retrieval_tool, finance_utility_tool]
    agent_executor = create_react_agent(llm, tools)
    
    return agent_executor, llm

def run_finance_agent(session_id: str, user_query: str, data_summary: str) -> FinanceAgentResponse:
    agent_executor, llm = get_agent()
    
    # Load history
    history = load_history(session_id)
    messages = []
    
    system_prompt = f"""You are an expert financial data analyst and AI advisor.
You have access to the user's detailed financial data summary below, which may include exact row-by-row transaction data.
When a user asks a question:
1. STRICTLY base your answer on the provided data.
2. Use your calculator tool for any math to ensure precision.
3. Be concise, professional, and precise.

Data Summary:
{data_summary}

Respond with the final answer clearly.
"""
    messages.append(SystemMessage(content=system_prompt))
    
    for msg in history[-10:]:  # Keep last 10 messages for context
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=user_query))
    
    # Run agent
    try:
        response_state = agent_executor.invoke({"messages": messages})
        final_message = response_state["messages"][-1].content
        
        # Extract tools used and sources by inspecting intermediate steps if available,
        # but LangGraph prebuilt agent stores tool messages in the state
        tools_used = []
        sources = set()
        
        for m in response_state["messages"]:
            if m.type == "tool":
                tools_used.append(m.name)
                if m.name == "retrieval_tool":
                    # Simple heuristic: extract sources from the tool output
                    # Output is formatted like "[Source: filename.pdf]\ncontent"
                    import re
                    matches = re.findall(r"\[Source: (.*?)\]", m.content)
                    for match in matches:
                        sources.add(match)
        
        # Log retrieval if happened
        if "retrieval_tool" in tools_used:
            log_retrieval(session_id, user_query, list(sources))
            
        # We now use the LLM to structure the output using the Pydantic model
        structured_llm = llm.with_structured_output(FinanceAgentResponse)
        
        struct_prompt = f"""
        Extract the following information from this interaction to populate the required JSON structure.
        
        User Question: {user_query}
        Final Answer: {final_message}
        Tools Used: {tools_used}
        Sources: {list(sources)}
        
        Provide the structured response. For confidence, estimate based on whether you had enough data (high, medium, low).
        Timestamp should be the current time.
        """
        
        structured_response = structured_llm.invoke([HumanMessage(content=struct_prompt)])
        
        # Ensure timestamp is set correctly if LLM hallucinates it
        structured_response.timestamp = datetime.now().isoformat()
        
        # Save to memory
        save_message(session_id, "user", user_query)
        save_message(session_id, "assistant", final_message)
        
        return structured_response
        
    except Exception as e:
        # Fallback in case of error
        print(f"Agent execution error: {e}")
        
        err_msg = f"I encountered an error while processing your request: {e}"
        save_message(session_id, "user", user_query)
        save_message(session_id, "assistant", err_msg)
        
        return FinanceAgentResponse(
            question=user_query,
            answer=err_msg,
            sources=[],
            tools_used=[],
            confidence="low",
            timestamp=datetime.now().isoformat()
        )
