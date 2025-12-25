from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
from pathlib import Path

class ResumeStore:
    """Store and search resumes using FAISS vector database"""
    
    def __init__(self, model="llama2"):
        self.embeddings = OllamaEmbeddings(model=model)
        self.vectorstore = None
        self.resumes = []
        self.store_path = Path("database/faiss_index")
        self.metadata_path = Path("database/resume_metadata.pkl")
    
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
        """Save FAISS index and metadata to disk"""
        if self.vectorstore is None:
            print("⚠️ No vector store to save")
            return
        
        self.store_path.parent.mkdir(exist_ok=True)
        
        # Save FAISS index
        self.vectorstore.save_local(str(self.store_path))
        
        # Save metadata separately
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.resumes, f)
        
        print(f"💾 Saved {len(self.resumes)} resumes to FAISS index")
    
    def searload(self):
        """Load FAISS index and metadata from disk"""
        if self.store_path.exists():
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(self.store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'rb') as f:
                    self.resumes = pickle.load(f)
            
            print(f"📂 Loaded {len(self.resumes)} resumes from FAISS index")