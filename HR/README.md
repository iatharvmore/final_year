# AI Resume Matcher 🚀

A professional HR tool to rank and analyze resumes against a job description using **Google Gemini AI**.

## Features
- **Intelligent Ranking**: Uses TF-IDF and Cosine Similarity to provide a baseline match score.
- **Deep Analysis**: Leverages Google Gemini (1.5 Flash/Pro) to extract Education, Experience, and Project details.
- **Support for Multiple Formats**: Processes both PDF and DOCX files.
- **Streamlit Frontend**: Clean, modern, and interactive user interface.
- **Excel Export**: Download the complete analysis as an Excel file for further use.

## Setup Instructions

1. **Clone the repository** (or navigate to the project directory).

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**:
   - Create a `.env` file from `.env.example`.
   - Add your [Google AI Studio (Gemini)](https://aistudio.google.com/) API key:
     ```
     GOOGLE_API_KEY=your_key_here
     ```
   - Alternatively, you can enter the API key directly in the Streamlit sidebar.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Technology Stack
- **Frontend**: Streamlit
- **LLM**: Google Gemini via LangChain
- **Text Processing**: Scikit-learn (TF-IDF), python-docx, pypdf
- **Data Handling**: Pandas, OpenPyXL
