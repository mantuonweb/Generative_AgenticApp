from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import json
from pathlib import Path

class ResumeStore:
    """Store and search resumes using FAISS vector database"""
    
    def __init__(self, model="llama2"):
        self.embeddings = OllamaEmbeddings(model=model)
        self.vectorstore = None
        self.resumes = []
        self.store_path = Path("data/resume_store.json")
    
    def add_resume(self, resume_data):
        """Add a resume to the store"""
        self.resumes.append(resume_data)
        
        # Create searchable text
        search_text = self._create_search_text(resume_data)
        
        # Create document
        doc = Document(
            page_content=search_text,
            metadata=resume_data
        )
        
        # Add to vector store
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents([doc], self.embeddings)
        else:
            self.vectorstore.add_documents([doc])
        
        print(f"✅ Added resume: {resume_data.get('name', 'Unknown')}")
    
    def _create_search_text(self, resume_data):
        """Create searchable text from resume data"""
        parts = [
            f"Name: {resume_data.get('name', '')}",
            f"Skills: {', '.join(resume_data.get('technical_skills', []))}",
            f"Tools: {', '.join(resume_data.get('tools', []))}",
            f"Soft Skills: {', '.join(resume_data.get('soft_skills', []))}",
            f"Experience: {resume_data.get('experience_years', '')}"
        ]
        return " | ".join(parts)
    
    def search(self, query, k=3):
        """Search for resumes matching the query"""
        if self.vectorstore is None:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.metadata for doc in results]
    
    def save(self):
        """Save resumes to disk"""
        self.store_path.parent.mkdir(exist_ok=True)
        with open(self.store_path, 'w') as f:
            json.dump(self.resumes, f, indent=2)
        print(f"💾 Saved {len(self.resumes)} resumes")
    
    def load(self):
        """Load resumes from disk"""
        if self.store_path.exists():
            with open(self.store_path, 'r') as f:
                self.resumes = json.load(f)
            
            # Rebuild vector store
            for resume in self.resumes:
                search_text = self._create_search_text(resume)
                doc = Document(page_content=search_text, metadata=resume)
                
                if self.vectorstore is None:
                    self.vectorstore = FAISS.from_documents([doc], self.embeddings)
                else:
                    self.vectorstore.add_documents([doc])
            
            print(f"📂 Loaded {len(self.resumes)} resumes")