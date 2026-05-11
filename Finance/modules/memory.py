import sqlite3
import json
import os
from datetime import datetime

DB_PATH = "Finance/data/finance_chat.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create chat history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    ''')
    
    # Create retrieval logs table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS retrieval_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        sources TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()

def load_history(session_id: str):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]

def log_retrieval(session_id: str, query: str, sources: list):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat()
    sources_json = json.dumps(sources)
    cursor.execute(
        "INSERT INTO retrieval_logs (session_id, query, sources, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, query, sources_json, timestamp)
    )
    conn.commit()
    conn.close()
