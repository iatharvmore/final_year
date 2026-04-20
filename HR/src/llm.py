import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LLMService:
    """Service for interacting with Google Gemini LLM models."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize the LLM service with the specified model.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        self.llm = None
        
    def initialize(self) -> bool:
        """Initialize connection to the Gemini LLM.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY not found in environment variables.")
            return False
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=0.1,
                convert_system_message_to_human=True
            )
            # Test connection
            self.llm.invoke("Hello")
            print(f"Successfully connected to Gemini with the {self.model_name} model.")
            return True
        except Exception as e:
            print(f"Error connecting to Gemini: {e}")
            return False
            
    def extract_education(self, resume_text: str) -> str:
        """Extract education details from resume text.
        
        Args:
            resume_text: The text content of a resume
            
        Returns:
            str: Extracted education information
        """
        if not self.llm:
            return "LLM not initialized"
            
        education_prompt = f"""
        TASK: Extract ONLY the education information from the resume text below.

        Include:
        - Degree names (Bachelor's, Master's, PhD, etc.)
        - Field of study (Engineering, Computer Science, etc.)
        - University/college/institution names.
        - Graduation years or dates

        If no education information is found, respond with "Not Found".

        Resume Text:
        {resume_text}

        Output ONLY the education details:
        """
        
        try:
            response = self.llm.invoke(education_prompt)
            education = response.content.strip()
            return education if education else "Not Found"
        except Exception as e:
            print(f"Error extracting education: {e}")
            return "Error during extraction"
    
    def extract_experience(self, resume_text: str) -> str:
        """Extract work experience details from resume text.
        
        Args:
            resume_text: The text content of a resume
            
        Returns:
            str: Extracted work experience information
        """
        if not self.llm:
            return "LLM not initialized"
            
        experience_prompt = f"""
        TASK: Extract details of work experience from the resume text below.

        Include for each experience:
        - Company name
        - Job title
        - Dates of employment
        - Responsibilities and achievements
        
        If no experience information is found, respond with "Not Found".
        Ensure the output is well-formatted and easy to read.

        Resume Text:
        {resume_text}

        Output ONLY the experience details:
        """
        
        try:
            response = self.llm.invoke(experience_prompt)
            experience = response.content.strip()
            return experience if experience else "Not Found"
        except Exception as e:
            print(f"Error extracting experience: {e}")
            return "Error during extraction"
    
    def extract_projects(self, resume_text: str) -> str:
        """Extract project details from resume text.
        
        Args:
            resume_text: The text content of a resume
            
        Returns:
            str: Extracted project information
        """
        if not self.llm:
            return "LLM not initialized"
            
        projects_prompt = f"""
        TASK: Extract project details from the resume text below. Include:
        - Project names
        - Roles (e.g., Team Lead, Developer)
        - Technologies or tools used
        - Achievements or outcomes

        Output the project details as a list or bullet points. If no project details are found, respond with "Not Found".
        
        Resume Text:
        {resume_text}
        
        Ensure the output is well-formatted and easy to read.
        """
        
        try:
            response = self.llm.invoke(projects_prompt)
            projects = response.content.strip()
            return projects if projects else "Not Found"
        except Exception as e:
            print(f"Error extracting projects: {e}")
            return "Error during extraction"