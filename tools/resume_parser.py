import PyPDF2
import docx
import json
from pathlib import Path

class ResumeParser:
    """Extract text from PDF and DOCX resumes"""
    
    def __init__(self):
        self.name = "resume_parser"
    
    def parse_pdf(self, file_path):
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def parse_docx(self, file_path):
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def parse(self, file_path):
        """Parse resume based on file type"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self.parse_pdf(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self.parse_docx(file_path)
        elif file_path.suffix.lower() == '.txt':
            return file_path.read_text()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")