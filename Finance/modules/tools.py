from langchain.tools import tool
import math
from Finance.modules.rag_engine import retrieve_context

@tool
def calculator_tool(expression: str) -> str:
    """
    Calculate mathematical expressions like CAGR, EMI, profit/loss, percentages.
    Example expressions: '100 * 1.05', '(500 - 400) / 400 * 100'.
    """
    try:
        # Use a safe subset of math and builtins for eval
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max, "pow": pow})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def retrieval_tool(query: str) -> str:
    """
    Search the uploaded financial documents for relevant information.
    Use this when the user asks about something that might be in an uploaded PDF, DOCX, or TXT file.
    """
    docs = retrieve_context(query)
    if not docs:
        return "No relevant information found in the uploaded documents."
    
    formatted_docs = []
    for d in docs:
        source = d.metadata.get("source", "Unknown")
        formatted_docs.append(f"[Source: {source}]\n{d.page_content}")
        
    return "\n\n".join(formatted_docs)

@tool
def finance_utility_tool(stat_name: str) -> str:
    """
    Fetch global statistics. Valid stat_names are:
    'summary' - Returns general instructions to view the dashboard summary.
    """
    return "To view full global statistics, please refer to the Dashboard & Analytics tab which contains the active data context."
