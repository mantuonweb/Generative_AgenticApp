import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from tools.resume_parser import ResumeParser
from agents.skill_extractor import SkillExtractorAgent
from agents.resume_store import ResumeStore
from agents.search_agent import SearchAgent
from pathlib import Path

class ResumeAgentSystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        print("🚀 Initializing Resume Agent System...")
        self.parser = ResumeParser()
        self.skill_extractor = SkillExtractorAgent()
        self.resume_store = ResumeStore()
        self.search_agent = SearchAgent(self.resume_store)
        print("✅ System ready!\n")
    
    def ingest_resume(self, file_path):
        """Ingest a single resume"""
        print(f"\n📄 Processing: {file_path}")
        
        # 1. Parse resume
        text = self.parser.parse(file_path)
        print(f"✅ Extracted {len(text)} characters")
        
        # 2. Extract skills
        print("🤖 Extracting skills...")
        skills_data = self.skill_extractor.extract_skills(text)
        skills_data['file_path'] = str(file_path)
        
        # 3. Store in vector DB
        self.resume_store.add_resume(skills_data)
        
        return skills_data
    
    def ingest_folder(self, folder_path):
        """Ingest all resumes from a folder"""
        folder = Path(folder_path)
        resume_files = list(folder.glob("*.pdf")) + list(folder.glob("*.docx")) + list(folder.glob("*.txt"))
        
        print(f"\n📁 Found {len(resume_files)} resumes")
        
        for file in resume_files:
            try:
                self.ingest_resume(file)
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        
        self.resume_store.save()
    
    def search_candidates(self, query):
        """Search for candidates"""
        print(f"\n🔍 Searching for: {query}")
        results = self.search_agent.search(query)
        
        print(f"\n📊 Found {len(results)} candidates:\n")
        
        for i, result in enumerate(results, 1):
            resume = result['resume']
            print(f"{i}. {resume.get('name', 'Unknown')}")
            print(f"   Skills: {', '.join(resume.get('technical_skills', [])[:5])}")
            print(f"   Match: {result['match_reason']}")
            print()
        
        return results
    
    def interactive_mode(self):
        """Interactive search mode"""
        print("\n" + "="*60)
        print("🤖 Resume Agent System - Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  search: <query>  - Search for candidates")
        print("  ingest: <path>   - Ingest resume or folder")
        print("  list             - List all resumes")
        print("  quit             - Exit")
        print("-"*60)
        
        while True:
            cmd = input("\n💬 You: ").strip()
            
            if cmd.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if cmd.lower().startswith('search:'):
                query = cmd[7:].strip()
                self.search_candidates(query)
            
            elif cmd.lower().startswith('ingest:'):
                path = cmd[7:].strip()
                if Path(path).is_dir():
                    self.ingest_folder(path)
                else:
                    self.ingest_resume(path)
            
            elif cmd.lower() == 'list':
                print(f"\n📋 Total resumes: {len(self.resume_store.resumes)}")
                for i, resume in enumerate(self.resume_store.resumes, 1):
                    print(f"{i}. {resume.get('name', 'Unknown')} - {len(resume.get('technical_skills', []))} skills")
            
            else:
                print("❌ Unknown command. Try 'search:', 'ingest:', 'list', or 'quit'")

def main():
    system = ResumeAgentSystem()
    
    # Load existing data
    system.resume_store.load()
    
    # Start interactive mode
    system.interactive_mode()

if __name__ == "__main__":
    main()