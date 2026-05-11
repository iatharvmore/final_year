from langchain_core.tools import tool
from .rag_utils import load_or_create_faiss, save_ticket
import json

@tool
def retrieval_tool(query: str) -> str:
    """
    Search the knowledge base and uploaded documents for answers to user questions.
    Use this tool when the user asks about platform features, policies, billing standard procedures, or technical issues.
    """
    vector_store = load_or_create_faiss()
    docs = vector_store.similarity_search(query, k=3)
    
    if not docs:
        return "No relevant information found in the knowledge base."
    
    results = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown Source")
        results.append(f"[Source: {source}]\n{doc.page_content}")
        
    return "\n\n".join(results)

@tool
def customer_query_tool(customer_id_or_email: str) -> str:
    """
    Fetch specific account, subscription, or invoice details for a customer.
    Use this tool when the user provides an account ID, email, or invoice number to look up their specific details.
    """
    # Mock implementation
    mock_data = {
        "status": "Active",
        "subscription_plan": "Pro",
        "last_invoice_status": "Paid",
        "recent_issues": "None"
    }
    return json.dumps(mock_data)

@tool
def ticket_utility_tool(issue_description: str, priority: str = "Medium") -> str:
    """
    Create a support ticket or escalate an issue to a human agent.
    Use this tool when the issue is too complex to resolve automatically, or when the user explicitly requests human escalation.
    """
    import random
    ticket_id = f"TCK-{random.randint(1000, 9999)}"
    
    # Save the ticket to the SQLite database
    # Determine basic sentiment from priority
    sentiment = "Angry" if priority.lower() == "high" or priority.lower() == "critical" else "Neutral"
    save_ticket(ticket_id, issue_description, status="Open", sentiment=sentiment)
    
    return f"Ticket {ticket_id} created successfully with priority {priority}. A human agent will review it shortly."

@tool
def csv_analytics_tool(sql_query: str) -> str:
    """
    Execute an SQL query to analyze the uploaded CSV data and return the results.
    The table is named 'uploaded_csv_data'. Use standard SQLite SQL syntax.
    If you do not know the column names, ALWAYS execute 'PRAGMA table_info(uploaded_csv_data);' first to get the schema before asking the user.
    Example queries:
    - 'SELECT COUNT(*) FROM uploaded_csv_data WHERE Status != "Closed"'
    - 'SELECT Status, COUNT(*) FROM uploaded_csv_data GROUP BY Status'
    Important: Always return ONLY the SQL query string.
    """
    import sqlite3
    import os
    from .rag_utils import DB_PATH
    
    if not os.path.exists(DB_PATH):
        return "No database found. Please upload a CSV first."
        
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if the table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='uploaded_csv_data'")
        if not c.fetchone():
            return "No CSV data has been uploaded yet."
            
        c.execute(sql_query)
        results = c.fetchall()
        
        # Get column names
        column_names = [description[0] for description in c.description]
        
        # Format results beautifully
        if not results:
            return "Query executed successfully, but returned 0 rows."
            
        formatted_results = f"Columns: {', '.join(column_names)}\n"
        for row in results[:50]: # Limit to 50 rows to avoid token limit issues
            formatted_results += str(row) + "\n"
            
        if len(results) > 50:
            formatted_results += f"... and {len(results) - 50} more rows."
            
        return formatted_results
    except Exception as e:
        return f"SQL Execution Error: {e}"
    finally:
        if 'conn' in locals():
            conn.close()

def get_tools():
    return [retrieval_tool, customer_query_tool, ticket_utility_tool, csv_analytics_tool]
