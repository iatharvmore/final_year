import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .llm import LLMService
from .tools import ResumeTools

class ResumeAgent:
    """Agent for processing and analyzing resumes against a job description."""
    
    def __init__(self, 
                 job_description: str, # resume_directory is removed from here
                 llm_model: str = "gemini-2.5-flash",
                 output_file: str = "resume_analysis_results.xlsx"):
        """Initialize the resume agent.
        
        Args:
            job_description: Text of the job description
            llm_model: Name of the LLM model to use
            output_file: Path to the output Excel file
        """
        # self.resume_directory = resume_directory # This line is removed
        self.job_description = job_description
        self.output_file = output_file
        self.llm_service = LLMService(model_name=llm_model)
        self.tools = ResumeTools()
        
    def initialize(self) -> bool:
        """Initialize the agent components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Initialize LLM service
        return self.llm_service.initialize()
        
    # The get_resume_files method is removed from here as it's no longer needed
    # def get_resume_files(self) -> List[str]:
    #     """Get list of resume files from the directory."""
    #     # ... (removed content) ...
        
    def process_resume(self, resume_path: str) -> Dict[str, Any]:
        """Process a single resume and extract relevant details.
        
        Args:
            resume_path: Path to the resume file
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted details
        """
        filename = os.path.basename(resume_path)
        
        print(f"Processing: {filename}")

        # Extract text based on file type
        if resume_path.lower().endswith(".pdf"):
            resume_text = self.tools.extract_text_from_pdf(resume_path)
            
            if not resume_text or resume_text.isspace():
                print(f"Warning: No text extracted from {filename}")
                return {"File": filename, "Rank": -1}

        elif resume_path.lower().endswith((".docx", ".doc")):
            resume_text = self.tools.extract_text_from_docx(resume_path)
            
            if not resume_text or resume_text.isspace():
                print(f"Warning: No text extracted from {filename}")
                return {"File": filename, "Rank": -1}

        else:
            print(f"Skipping unsupported file type: {filename}")
            return {"File": filename, "Rank": -1}

        # Extract entities using LLM
        education = self.llm_service.extract_education(resume_text)
        experience = self.llm_service.extract_experience(resume_text)
        projects = self.llm_service.extract_projects(resume_text)

        # Calculate cosine similarity with JD
        similarity_score = self.tools.calculate_cosine_similarity(
            self.job_description, resume_text
        )

        return {
            'File': filename,
            'Education': education,
            'Experience': experience,
            'Projects': projects,
            'Cosine Similarity': similarity_score,
            'Rank': -1
        }
        
    def analyze_resumes(self, resume_paths: List[str], max_workers: int = 3) -> List[Dict[str, Any]]:
        """Process all resume files and rank them.
        
        Args:
            resume_paths: List of paths to the resume files (temporary files)
            max_workers: Maximum number of worker threads
            
        Returns:
            List[Dict[str, Any]]: List of ranked resume details
        """
        if not resume_paths:
            print("No valid resumes provided for analysis.")
            return []
            
        print(f"Found {len(resume_paths)} resumes. Starting analysis...")
        
        # Process resumes concurrently
        raw_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_resume, path) for path in resume_paths]
            
            for future in tqdm(futures):
                raw_results.append(future.result())
                
        # Rank resumes based on cosine similarity
        ranked_resumes = self.tools.rank_resumes(raw_results)
        
        return ranked_resumes
    
    def save_results(self, results: List[Dict[str, Any]]) -> bool:
        """Save results to Excel file.
        
        Args:
            results: List of ranked resume details
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        return self.tools.save_results_to_excel(results, self.output_file)