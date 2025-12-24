from langchain_ollama import OllamaLLM
import json
import re

class SkillExtractorAgent:
    """Agent that extracts skills from resume text"""
    
    def __init__(self, model="llama2"):
        self.llm = OllamaLLM(model=model)
        self.name = "SkillExtractor"
    
    def extract_skills(self, resume_text):
        """Extract skills from resume text using LLM"""
        
        prompt = f"""
You are a skill extraction expert. Extract ALL technical skills, soft skills, and tools from this resume.

Resume:
{resume_text[:2000]}  

Return ONLY a JSON object with this structure:
{{
    "name": "candidate name",
    "email": "email if found",
    "technical_skills": ["skill1", "skill2"],
    "soft_skills": ["skill1", "skill2"],
    "tools": ["tool1", "tool2"],
    "experience_years": "X years or estimate"
}}

JSON:
"""
        
        response = self.llm.invoke(prompt)
        
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                skills_data = json.loads(json_match.group())
            else:
                # Fallback: create basic structure
                skills_data = self._fallback_extraction(resume_text)
            
            return skills_data
        except:
            return self._fallback_extraction(resume_text)
    
    def _fallback_extraction(self, text):
        """Simple fallback extraction"""
        return {
            "name": "Unknown",
            "email": "",
            "technical_skills": [],
            "soft_skills": [],
            "tools": [],
            "experience_years": "Unknown"
        }