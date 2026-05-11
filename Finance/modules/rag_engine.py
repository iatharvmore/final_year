import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = "Finance/data/faiss_index"

def get_embeddings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

def process_and_index_document(uploaded_file):
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load based on file extension
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.pdf':
        loader = PyPDFLoader(temp_file_path)
    elif ext == '.docx':
        loader = Docx2txtLoader(temp_file_path)
    elif ext == '.txt':
        loader = TextLoader(temp_file_path)
    else:
        os.remove(temp_file_path)
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()
    
    # Add metadata for tracking
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        
    os.remove(temp_file_path)

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Create or update FAISS index
    embeddings = get_embeddings()
    
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        except Exception as e:
            print(f"Error loading existing index, creating new one: {e}")
            vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
    # Save local
    vectorstore.save_local(FAISS_INDEX_PATH)
    return len(chunks)

def retrieve_context(query: str, k: int = 3):
    if not os.path.exists(FAISS_INDEX_PATH):
        return []
        
    embeddings = get_embeddings()
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error retrieving from FAISS: {e}")
        return []
