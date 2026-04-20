import os
import pandas as pd
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:
    from docx import Document
except ImportError:
    print("python-docx not installed. Please install it using 'pip install python-docx' for .docx support.")
    Document = None

try:
    import pypdf
except ImportError:
    print("pypdf not installed. Please install it using 'pip install pypdf' for .pdf support.")
    pypdf = None


class ResumeTools:
    """A collection of utility tools for resume processing."""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file."""
        if pypdf is None:
            return ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extracts text from a DOCX file."""
        if Document is None:
            return ""
        try:
            document = Document(docx_path)
            text = ""
            for paragraph in document.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculates the cosine similarity between two text documents."""
        if not text1 or not text2:
            return 0.0

        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        vectors = vectorizer.toarray()
        
        
        if vectors.shape[0] < 2 or vectors.sum() == 0:
            return 0.0

        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)

    def rank_resumes(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ranks resumes based on cosine similarity."""
   
        valid_results = [r for r in results if r.get('Rank', -1) != -1]

      
        ranked_resumes = sorted(valid_results, key=lambda x: x.get('Cosine Similarity', 0), reverse=True)

        for i, resume in enumerate(ranked_resumes):
            resume['Rank'] = i + 1
        
        error_resumes = [r for r in results if r.get('Rank', -1) == -1]
        
        return ranked_resumes + error_resumes 

    def save_results_to_excel(self, results: List[Dict[str, Any]], output_file: str) -> bool:
        """Saves the analysis results to an Excel file."""
        if not results:
            print("No results to save.")
            return False
        
        try:
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            print(f"Results saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results to Excel: {e}")
            return False