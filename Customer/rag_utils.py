import os
import sqlite3
import json
import uuid
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "customer_chat_history.db")
FAISS_PATH = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT,
            customer TEXT,
            issue TEXT,
            status TEXT,
            sentiment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
    conn.commit()
    conn.close()

def load_chat_history(session_id: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC', (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def clear_chat_history(session_id: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM chat_history WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

def save_ticket(ticket_id: str, issue: str, status: str = "Open", sentiment: str = "Neutral"):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Assume 'Customer' is 'User' since it's from chat
    c.execute('INSERT INTO tickets (ticket_id, customer, issue, status, sentiment) VALUES (?, ?, ?, ?, ?)', 
              (ticket_id, "User", issue, status, sentiment))
    conn.commit()
    conn.close()

def load_tickets():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT ticket_id, customer, issue, status, sentiment FROM tickets ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return [{"TicketID": row[0], "Customer": row[1], "Issue": row[2], "Status": row[3], "Sentiment": row[4]} for row in rows]

def get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)

def load_or_create_faiss():
    embeddings = get_embeddings()
    if os.path.exists(FAISS_PATH) and os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        try:
            return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    
    # Initialize empty FAISS index
    return FAISS.from_texts(["Initial empty document"], embeddings)

def save_faiss(vector_store):
    os.makedirs(FAISS_PATH, exist_ok=True)
    vector_store.save_local(FAISS_PATH)

def process_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    vector_store = load_or_create_faiss()
    vector_store.add_documents(docs)
    save_faiss(vector_store)
    return len(docs)

def process_csv_to_sql(csv_file):
    import pandas as pd
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_csv(csv_file)
        # We replace the table so fresh uploads override the old analytics table
        df.to_sql("uploaded_csv_data", conn, if_exists="replace", index=False)
        return True
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return False
    finally:
        conn.close()

def index_knowledge_base():
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    if not os.path.exists(kb_path):
        return 0
    
    with open(kb_path, "r") as f:
        kb_data = json.load(f)
    
    docs = []
    for entry in kb_data.get("knowledge_base", []):
        content = f"Topic: {entry.get('topic')}\nQuery/Intent: {entry.get('query')}\nStandard Response: {entry.get('standard_response')}"
        docs.append(Document(page_content=content, metadata={"source": "knowledge_base.json", "topic": entry.get('topic')}))
    
    if docs:
        return process_documents(docs)
    return 0
