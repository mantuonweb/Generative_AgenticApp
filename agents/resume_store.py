from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pickle
from pathlib import Path
import hashlib
import json

class ResumeStore:
    """Store and search resumes using FAISS vector database"""
    
    def __init__(self, model="llama2"):
        self.embeddings = OllamaEmbeddings(model=model)
        self.vectorstore = None
        self.resumes = []
        self.resume_hashes = set()  # Track unique resumes
        self.store_path = Path("db/faiss_index.pkl")
        self.metadata_path = Path("db/resume_metadata.pkl")
        self.hashes_path = Path("db/resume_hashes.pkl")
    
    def _generate_hash(self, resume_data):
        """Generate a unique hash for a resume based on key fields"""
        # Create a deterministic string from key resume fields
        key_fields = {
            'name': resume_data.get('name', ''),
            'email': resume_data.get('email', ''),
            'phone': resume_data.get('phone', ''),
            'technical_skills': sorted(resume_data.get('technical_skills', [])),
            'experience_years': resume_data.get('experience_years', '')
        }
        
        # Convert to JSON string and hash it
        resume_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.md5(resume_str.encode()).hexdigest()
    
    def add_resume(self, resume_data):
        """Add a resume to the store (prevents duplicates)"""
        # Generate hash for this resume
        resume_hash = self._generate_hash(resume_data)
        
        # Check if already exists
        if resume_hash in self.resume_hashes:
            print(f"⚠️ Skipping duplicate resume: {resume_data.get('name', 'Unknown')}")
            return False
        
        # Add to tracking set
        self.resume_hashes.add(resume_hash)
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
        return True
    
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
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.resumes, f)
        
        # Save hashes
        with open(self.hashes_path, 'wb') as f:
            pickle.dump(self.resume_hashes, f)
        
        print(f"💾 Saved {len(self.resumes)} resumes to FAISS index")
    
    def load(self):
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
            
            # Load hashes
            if self.hashes_path.exists():
                with open(self.hashes_path, 'rb') as f:
                    self.resume_hashes = pickle.load(f)
            
            print(f"📂 Loaded {len(self.resumes)} resumes from FAISS index")
    
    def clear(self):
        """Clear all stored resumes and start fresh"""
        self.vectorstore = None
        self.resumes = []
        self.resume_hashes = set()
        
        # Remove files if they exist
        if self.store_path.exists():
            import shutil
            shutil.rmtree(self.store_path)
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        if self.hashes_path.exists():
            self.hashes_path.unlink()
        
        print("🗑️ Cleared all resume data")
    
    def get_stats(self):
        """Get statistics about stored resumes"""
        return {
            'total_resumes': len(self.resumes),
            'unique_hashes': len(self.resume_hashes)
        }