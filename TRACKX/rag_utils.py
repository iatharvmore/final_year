import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "data", "faiss_index")

from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

class LocalOfflineEmbeddings(Embeddings):
    def __init__(self):
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
    return LocalOfflineEmbeddings()

def process_and_index_performance_doc(uploaded_file):
    """Saves and indexes a uploaded file to the TrackX FAISS index."""
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

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
    
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name
        
    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = get_embeddings()
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            vectorstore.add_documents(chunks)
        except Exception as e:
            print(f"Error loading existing index, creating new one: {e}")
            vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
    vectorstore.save_local(FAISS_INDEX_PATH)
    return len(chunks)

def retrieve_trackx_context(query: str, k: int = 3):
    """Retrieves standard benchmarks, performance objectives, or general standards from FAISS."""
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return []
        
    embeddings = get_embeddings()
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error retrieving from TrackX FAISS: {e}")
        return []
