from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import numpy as np

class SearchAgent:
    """Agent that interprets search queries and finds candidates"""
    
    def __init__(self, resume_store, model="llama2"):
        self.llm = OllamaLLM(model=model)
        self.resume_store = resume_store
        self.name = "SearchAgent"
        
        # Initialize embedding model for semantic similarity
        print("🔄 Loading semantic similarity model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Semantic model loaded")
    
    def search(self, query):
        """Search for candidates based on natural language query"""
        
        # Extract required skills from query
        required_skills = self._extract_required_skills(query)
        print(f"🔍 Required skills extracted: {required_skills}")
        
        # Create enhanced query for vector search
        enhanced_query = " ".join(required_skills)
        
        # Search in vector store
        results = self.resume_store.search(enhanced_query, k=3)
        
        # Rank and explain results with percentage matching
        ranked_results = self._rank_results(query, results, required_skills)
        
        return ranked_results
    
    def _extract_required_skills(self, query):
        """Extract core required skills from query using LLM"""
        prompt = f"""
Extract the technical skills from this query: "{query}"

Return ONLY a comma-separated list of skills. No explanations.

Output:"""
        
        response = self.llm.invoke(prompt).strip()
        
        # Clean up the response
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.lower().startswith(('sure', 'here', 'based', 'the ', 'essential')):
                continue
            if ',' in line or len(line.split()) <= 5:
                response = line
                break
        
        response = response.rstrip('.!:')
        skills = [skill.strip().lower() for skill in response.split(',')]
        return [s for s in skills if s and len(s) > 1 and not s.startswith(('sure', 'here', 'based'))]
    
    def _rank_results(self, original_query, results, required_skills):
        """Rank and explain why candidates match with percentage matching"""
        ranked = []
        
        for resume in results:
            candidate_skills = resume.get('technical_skills', [])
            
            print(f"\n📋 Analyzing: {resume.get('name', 'Unknown')}")
            print(f"   Candidate skills: {candidate_skills}")
            
            # Use hybrid matching: direct + semantic similarity
            direct_matches = self._find_direct_matches(required_skills, candidate_skills)
            print(f"   Direct matches: {direct_matches}")
            
            # For unmatched skills, use embedding similarity
            remaining_required = [s for s in required_skills if s not in direct_matches]
            print(f"   Remaining to match: {remaining_required}")
            
            semantic_matches = []
            semantic_explanations = {}
            
            if remaining_required:
                for req_skill in remaining_required:
                    match_result = self._find_semantic_match_embedding(req_skill, candidate_skills)
                    print(f"   Semantic check '{req_skill}': {match_result}")
                    if match_result['is_match']:
                        semantic_matches.append(req_skill)
                        semantic_explanations[req_skill] = match_result['explanation']
            
            all_matches = direct_matches + semantic_matches
            exact_count = len(direct_matches)
            
            print(f"   ✅ Total matches: {all_matches}")
            
            numerical_match = self._calculate_numerical_match(
                required_skills, 
                candidate_skills, 
                all_matches,
                exact_count
            )
            
            final_explanation = self._create_honest_explanation(
                direct_matches,
                semantic_matches,
                semantic_explanations,
                required_skills,
                candidate_skills,
                numerical_match
            )
            
            ranked.append({
                'resume': resume,
                'score': numerical_match['overall_percentage'],
                'match_percentage': numerical_match['overall_percentage'],
                'llm_match_percentage': numerical_match['llm_match_percentage'],
                'numerical_match_percentage': numerical_match['exact_match_percentage'],
                'matched_skills': all_matches,
                'required_skills': required_skills,
                'total_candidate_skills': len(candidate_skills),
                'match_details': numerical_match,
                'match_reason': final_explanation
            })
        
        return sorted(ranked, key=lambda x: x['score'], reverse=True)
    
    def _find_direct_matches(self, required_skills, candidate_skills):
        """Find direct string matches (case-insensitive) - STRICT"""
        candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
        direct_matches = []
        
        for req_skill in required_skills:
            req_lower = req_skill.lower().strip()
            
            # Only exact match or if required skill is substring of candidate skill
            # NOT the other way around (prevents "java" matching "javascript")
            for cand_skill in candidate_skills_lower:
                # Exact match
                if req_lower == cand_skill:
                    direct_matches.append(req_skill)
                    break
                # Required skill is contained in candidate skill (e.g., "script" in "javascript")
                # But NOT if it's just a partial word match
                elif req_lower in cand_skill and len(req_lower) > 3:
                    # Additional check: make sure it's a meaningful match
                    # "java" should NOT match "javascript" 
                    if req_lower == "java" and cand_skill == "javascript":
                        continue  # Skip this false match
                    direct_matches.append(req_skill)
                    break
        
        return direct_matches
    
    def _find_semantic_match_embedding(self, required_skill, candidate_skills, threshold=0.65):
        """Use embeddings to find semantic matches - NO HALLUCINATION!"""
        
        if not candidate_skills:
            return {
                'is_match': False,
                'matched_skill': None,
                'similarity': 0.0,
                'explanation': "No candidate skills"
            }
        
        # Encode required skill
        req_embedding = self.embedding_model.encode([required_skill])[0]
        
        # Encode all candidate skills
        cand_embeddings = self.embedding_model.encode(candidate_skills)
        
        # Calculate cosine similarity
        similarities = []
        for i, cand_emb in enumerate(cand_embeddings):
            # Cosine similarity
            similarity = np.dot(req_embedding, cand_emb) / (
                np.linalg.norm(req_embedding) * np.linalg.norm(cand_emb)
            )
            similarities.append((candidate_skills[i], similarity))
        
        # Find best match
        best_match = max(similarities, key=lambda x: x[1])
        best_skill, best_score = best_match
        
        # Only accept if similarity is above threshold
        if best_score >= threshold:
            return {
                'is_match': True,
                'matched_skill': best_skill,
                'similarity': round(best_score, 3),
                'explanation': f"{best_skill} ({round(best_score * 100, 1)}% similar)"
            }
        
        return {
            'is_match': False,
            'matched_skill': None,
            'similarity': round(best_score, 3),
            'explanation': f"Best: {best_skill} only {round(best_score * 100, 1)}% similar"
        }
    
    def _calculate_numerical_match(self, required_skills, candidate_skills, all_matches, exact_count):
        """Calculate numerical match percentages"""
        exact_match_percentage = (exact_count / len(required_skills) * 100) if required_skills else 0
        total_match_count = len(all_matches)
        llm_match_percentage = (total_match_count / len(required_skills) * 100) if required_skills else 0
        
        llm_match_percentage = min(llm_match_percentage, 100.0)
        exact_match_percentage = min(exact_match_percentage, 100.0)
        
        # Weighted: 70% exact, 30% semantic
        overall_percentage = (exact_match_percentage * 0.7) + (llm_match_percentage * 0.3)
        
        return {
            'exact_matches': exact_count,
            'exact_match_percentage': round(exact_match_percentage, 2),
            'llm_matches': total_match_count,
            'llm_match_percentage': round(llm_match_percentage, 2),
            'overall_percentage': round(overall_percentage, 2),
            'required_count': len(required_skills),
            'candidate_count': len(candidate_skills)
        }
    
    def _create_honest_explanation(self, direct_matches, semantic_matches, semantic_explanations, 
                                   required_skills, candidate_skills, numerical_match):
        """Create an honest explanation"""
        
        parts = []
        
        if direct_matches:
            parts.append(f"Exact: {', '.join(direct_matches)}")
        
        if semantic_matches:
            semantic_details = [semantic_explanations.get(s, s) for s in semantic_matches]
            parts.append(f"Similar: {'; '.join(semantic_details)}")
        
        missing = [r for r in required_skills if r not in direct_matches and r not in semantic_matches]
        if missing:
            parts.append(f"Missing: {', '.join(missing)}")
        
        match_summary = f"{len(direct_matches + semantic_matches)}/{len(required_skills)} match"
        
        if not parts:
            return f"No match | Candidate: {', '.join(candidate_skills[:5])} | {numerical_match['overall_percentage']}%"
        
        return f"{match_summary} | {' | '.join(parts)} | {numerical_match['overall_percentage']}%"