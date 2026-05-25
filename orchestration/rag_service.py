import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAISS_HR_PATH = os.path.join(DATA_DIR, "faiss_hr")
FAISS_CUSTOMER_PATH = os.path.join(DATA_DIR, "faiss_customer")
FAISS_FINANCE_PATH = os.path.join(DATA_DIR, "faiss_finance")

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

class LocalOfflineEmbeddings(Embeddings):
    def __init__(self):
        # A vocabulary dimension of 512 features is excellent for lightweight local search
        self.vectorizer = TfidfVectorizer(max_features=512, stop_words='english')
        self._fitted = False
        self.vectorizer.fit(["employee performance budget variance support ticket resume experience candidate client sales"])
        
    def fit_on_texts(self, texts):
        if texts:
            self.vectorizer.fit(texts)
            self._fitted = True
            
    def embed_documents(self, texts):
        if not self._fitted and texts:
            self.fit_on_texts(texts)
        vectors = self.vectorizer.transform(texts).toarray()
        return [v.tolist() for v in vectors]
        
    def embed_query(self, text):
        vectors = self.vectorizer.transform([text]).toarray()
        return vectors[0].tolist()

def get_embeddings():
    # Return local custom offline TF-IDF embeddings to bypass all quota/rate-limits and pytorch install issues
    return LocalOfflineEmbeddings()

def load_or_create_index(index_path, documents=None):
    embeddings = get_embeddings()
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        try:
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index at {index_path}: {e}")
    
    if documents:
        os.makedirs(index_path, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        return vector_store
    return None

def build_hr_documents():
    csv_path = "resume_dataset_2.csv"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    # limit to first 100 rows to prevent massive API overhead while retaining rich semantic diversity
    df_sample = df.head(100)
    
    docs = []
    for _, row in df_sample.iterrows():
        content = f"""Candidate Name: {row.get('Name')}
Email: {row.get('Email')}
Phone: {row.get('Phone')}
University: {row.get('University')}
Graduation Year: {row.get('Graduation_Year')}
Years of Experience: {row.get('Years_Experience')}
Job Role: {row.get('Job_Role')}
Skills: {row.get('Skills')}
Resume Text Summary: {str(row.get('Resume_Text'))[:1500]}"""
        
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "resume_dataset_2.csv",
                "candidate": str(row.get('Name')),
                "skills": str(row.get('Skills'))
            }
        ))
    return docs

def build_customer_documents():
    csv_path = "customer_support_tickets_120.csv"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    df_sample = df.head(120)
    
    docs = []
    for _, row in df_sample.iterrows():
        # Read the columns depending on actual file header structure
        # Standardizing headers internally
        row_dict = {str(k).lower().strip(): v for k, v in row.items()}
        ticket_id = row_dict.get('ticket_id') or row_dict.get('transaction_id') or "N/A"
        customer = row_dict.get('customer') or row_dict.get('department') or "N/A"
        issue = row_dict.get('issue') or row_dict.get('expense_type') or "N/A"
        status = row_dict.get('status') or "Open"
        sentiment = row_dict.get('sentiment') or "Neutral"
        
        content = f"""Ticket ID: {ticket_id}
Customer: {customer}
Issue Details: {issue}
Status: {status}
Sentiment Analysis: {sentiment}"""
        
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "customer_support_tickets_120.csv",
                "ticket_id": str(ticket_id)
            }
        ))
    return docs

def build_finance_documents():
    csv_path = "corporate_financial_analytics_data.csv"
    if not os.path.exists(csv_path):
        return []
    
    df = pd.read_csv(csv_path)
    df_sample = df.head(120)
    
    docs = []
    for _, row in df_sample.iterrows():
        row_dict = {str(k).lower().strip(): v for k, v in row.items()}
        tx_id = row_dict.get('transaction_id') or "N/A"
        date = row_dict.get('date') or "N/A"
        dept = row_dict.get('department') or "N/A"
        exp_type = row_dict.get('expense_type') or "N/A"
        vendor = row_dict.get('vendor') or "N/A"
        expense = row_dict.get('expense') or 0.0
        budget = row_dict.get('budget') or 0.0
        variance = row_dict.get('expense_variance') or 0.0
        status = row_dict.get('expense_status') or "Within Budget"
        
        content = f"""Transaction ID: {tx_id}
Date: {date}
Department: {dept}
Expense Type: {exp_type}
Vendor: {vendor}
Expense: ${expense}
Budget: ${budget}
Variance: ${variance}
Expense Status: {status}"""
        
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "corporate_financial_analytics_data.csv",
                "transaction_id": str(tx_id),
                "department": str(dept)
            }
        ))
    return docs

def initialize_rag_databases(force_rebuild=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. HR Index
    if force_rebuild or not os.path.exists(os.path.join(FAISS_HR_PATH, "index.faiss")):
        docs = build_hr_documents()
        if docs:
            print("Building HR Central FAISS index...")
            load_or_create_index(FAISS_HR_PATH, docs)
            
    # 2. Customer Index
    if force_rebuild or not os.path.exists(os.path.join(FAISS_CUSTOMER_PATH, "index.faiss")):
        docs = build_customer_documents()
        if docs:
            print("Building Customer Central FAISS index...")
            load_or_create_index(FAISS_CUSTOMER_PATH, docs)
            
    # 3. Finance Index
    if force_rebuild or not os.path.exists(os.path.join(FAISS_FINANCE_PATH, "index.faiss")):
        docs = build_finance_documents()
        if docs:
            print("Building Finance Central FAISS index...")
            load_or_create_index(FAISS_FINANCE_PATH, docs)

def query_rag_database(agent_name, query, k=4):
    """Retrieve relevant context for a specific L2 Agent."""
    embeddings = get_embeddings()
    
    if agent_name.upper() == "HR":
        index_path = FAISS_HR_PATH
    elif agent_name.upper() == "CUSTOMER":
        index_path = FAISS_CUSTOMER_PATH
    elif agent_name.upper() == "FINANCE":
        index_path = FAISS_FINANCE_PATH
    else:
        return []
        
    if not os.path.exists(index_path):
        # Trigger initialization if missing
        initialize_rag_databases()
        
    try:
        if os.path.exists(os.path.join(index_path, "index.faiss")):
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            docs = vectorstore.similarity_search(query, k=k)
            return docs
    except (AssertionError, Exception) as e:
        print(f"Error querying Central FAISS RAG ({agent_name}): {e}. Clearing and force-rebuilding central indexes...")
        try:
            import shutil
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
            initialize_rag_databases(force_rebuild=True)
            # Try once more
            if os.path.exists(os.path.join(index_path, "index.faiss")):
                vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                return vectorstore.similarity_search(query, k=k)
        except Exception as rebuild_err:
            print(f"Failed to rebuild index: {rebuild_err}")
        
    return []
