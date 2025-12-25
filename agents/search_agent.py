from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import numpy as np

class SearchAgent:
    """Agent that interprets search queries and finds candidates"""
    
    def __init__(self, resume_store, model="llama2", semantic_threshold=0.50):
        self.llm = OllamaLLM(model=model)
        self.resume_store = resume_store
        self.name = "SearchAgent"
        self.semantic_threshold = semantic_threshold
        
        # Initialize embedding model for semantic similarity
        print("🔄 Loading semantic similarity model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Semantic model loaded")
    
    def search(self, query):
        """Search for candidates based on natural language query"""
        
        # Extract required skills from query
        required_skills = self._extract_required_skills(query)
        print(f"🔍 Required skills extracted: {required_skills}")
        
        # Dynamically expand skills with LLM-generated relationships
        expanded_skills = self._expand_skills_dynamically(required_skills)
        if expanded_skills != required_skills:
            print(f"🔗 Expanded to include related skills: {expanded_skills}")
        
        # Create enhanced query for vector search
        enhanced_query = " ".join(expanded_skills)
        
        # Search in vector store (get more results for better ranking)
        results = self.resume_store.search(enhanced_query, k=5)
        
        # Rank and explain results with percentage matching
        ranked_results = self._rank_results(query, results, required_skills)
        
        # Return top 3
        return ranked_results[:3]
    
    def _expand_skills_dynamically(self, required_skills):
        """Dynamically expand required skills using LLM"""
        if not required_skills:
            return required_skills
        
        skills_str = ", ".join(required_skills)
        
        prompt = f"""For these technical skills: {skills_str}

List related/complementary technologies that are commonly used together.
Return ONLY a comma-separated list. No explanations.

Example: If input is "React", output might be: React, JavaScript, HTML, CSS, JSX, Redux

Output:"""
        
        try:
            response = self.llm.invoke(prompt).strip()
            
            # Clean up response
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if not line or line.lower().startswith(('sure', 'here', 'based', 'the ', 'for ')):
                    continue
                if ',' in line:
                    response = line
                    break
            
            response = response.rstrip('.!:')
            expanded = [skill.strip().lower() for skill in response.split(',')]
            expanded = [s for s in expanded if s and len(s) > 1]
            
            # Combine with original skills
            all_skills = list(set(required_skills + expanded))
            return all_skills
            
        except Exception as e:
            print(f"⚠️ Error expanding skills: {e}")
            return required_skills
    
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
            
            # Use hybrid matching: direct + semantic similarity + relationship bonus
            direct_matches = self._find_direct_matches(required_skills, candidate_skills)
            print(f"   Direct matches: {direct_matches}")
            
            # Check for relationship-based matches using embeddings
            relationship_matches = self._find_relationship_matches_dynamic(
                required_skills, candidate_skills, direct_matches
            )
            print(f"   Relationship matches: {relationship_matches}")
            
            # For unmatched skills, use embedding similarity
            already_matched = set(direct_matches + list(relationship_matches.keys()))
            remaining_required = [s for s in required_skills if s not in already_matched]
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
            
            all_matches = direct_matches + list(relationship_matches.keys()) + semantic_matches
            exact_count = len(direct_matches)
            relationship_count = len(relationship_matches)
            
            print(f"   ✅ Total matches: {all_matches}")
            
            numerical_match = self._calculate_numerical_match(
                required_skills, 
                candidate_skills, 
                all_matches,
                exact_count,
                relationship_count
            )
            
            final_explanation = self._create_honest_explanation(
                direct_matches,
                relationship_matches,
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
            
            for cand_skill in candidate_skills_lower:
                # Exact match
                if req_lower == cand_skill:
                    direct_matches.append(req_skill)
                    break
                # Required skill is contained in candidate skill
                elif req_lower in cand_skill and len(req_lower) > 3:
                    # Prevent "java" matching "javascript"
                    if req_lower == "java" and cand_skill == "javascript":
                        continue
                    direct_matches.append(req_skill)
                    break
        
        return direct_matches
    
    def _find_relationship_matches_dynamic(self, required_skills, candidate_skills, direct_matches):
        """Find matches based on semantic relationships using embeddings"""
        relationship_matches = {}
        candidate_skills_lower = [skill.lower().strip() for skill in candidate_skills]
        
        # Higher threshold for relationship matching (0.60-0.75 range)
        relationship_threshold = 0.60
        
        for req_skill in required_skills:
            # Skip if already directly matched
            if req_skill in direct_matches:
                continue
            
            req_lower = req_skill.lower()
            
            # Encode required skill
            req_embedding = self.embedding_model.encode([req_lower])[0]
            
            # Check similarity with all candidate skills
            for cand_skill in candidate_skills:
                cand_lower = cand_skill.lower().strip()
                
                # Encode candidate skill
                cand_embedding = self.embedding_model.encode([cand_lower])[0]
                
                # Calculate cosine similarity
                similarity = np.dot(req_embedding, cand_embedding) / (
                    np.linalg.norm(req_embedding) * np.linalg.norm(cand_embedding)
                )
                
                # If similarity is in the "related" range (not exact, but related)
                if relationship_threshold <= similarity < 0.85:
                    relationship_matches[req_skill] = {
                        'matched_skill': cand_skill,
                        'similarity': round(float(similarity), 3),
                        'explanation': f"{cand_skill} ({round(float(similarity) * 100, 1)}% related)"
                    }
                    break
        
        return relationship_matches
    
    def _find_semantic_match_embedding(self, required_skill, candidate_skills, threshold=None):
        """Use embeddings to find semantic matches - NO HALLUCINATION!"""
        
        if threshold is None:
            threshold = self.semantic_threshold
        
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
                'similarity': round(float(best_score), 3),
                'explanation': f"{best_skill} ({round(float(best_score) * 100, 1)}% similar)"
            }
        
        return {
            'is_match': False,
            'matched_skill': None,
            'similarity': round(float(best_score), 3),
            'explanation': f"Best: {best_skill} only {round(float(best_score) * 100, 1)}% similar"
        }
    
    def _calculate_numerical_match(self, required_skills, candidate_skills, all_matches, 
                                   exact_count, relationship_count):
        """Calculate numerical match percentages"""
        exact_match_percentage = (exact_count / len(required_skills) * 100) if required_skills else 0
        
        # Relationship matches count as 70% of exact match
        relationship_score = relationship_count * 0.7
        total_match_count = len(all_matches)
        
        # Calculate weighted match percentage
        semantic_count = total_match_count - exact_count - relationship_count
        weighted_matches = exact_count + relationship_score + (semantic_count * 0.5)
        llm_match_percentage = (weighted_matches / len(required_skills) * 100) if required_skills else 0
        
        llm_match_percentage = min(llm_match_percentage, 100.0)
        exact_match_percentage = min(exact_match_percentage, 100.0)
        
        # Weighted: 60% exact, 40% combined (relationship + semantic)
        overall_percentage = (exact_match_percentage * 0.6) + (llm_match_percentage * 0.4)
        
        return {
            'exact_matches': exact_count,
            'relationship_matches': relationship_count,
            'exact_match_percentage': round(exact_match_percentage, 2),
            'llm_matches': total_match_count,
            'llm_match_percentage': round(llm_match_percentage, 2),
            'overall_percentage': round(overall_percentage, 2),
            'required_count': len(required_skills),
            'candidate_count': len(candidate_skills)
        }
    
    def _create_honest_explanation(self, direct_matches, relationship_matches, semantic_matches, 
                                   semantic_explanations, required_skills, candidate_skills, 
                                   numerical_match):
        """Create an honest explanation"""
        
        parts = []
        
        if direct_matches:
            parts.append(f"✓ Exact: {', '.join(direct_matches)}")
        
        if relationship_matches:
            rel_details = [f"{k}→{v['matched_skill']}" for k, v in relationship_matches.items()]
            parts.append(f"≈ Related: {', '.join(rel_details)}")
        
        if semantic_matches:
            semantic_details = [semantic_explanations.get(s, s) for s in semantic_matches]
            parts.append(f"~ Similar: {'; '.join(semantic_details)}")
        
        missing = [r for r in required_skills 
                  if r not in direct_matches 
                  and r not in relationship_matches 
                  and r not in semantic_matches]
        if missing:
            parts.append(f"✗ Missing: {', '.join(missing)}")
        
        match_summary = f"{len(direct_matches + list(relationship_matches.keys()) + semantic_matches)}/{len(required_skills)} skills"
        
        if not parts:
            return f"No match | Candidate: {', '.join(candidate_skills[:5])} | {numerical_match['overall_percentage']}%"
        
        return f"{match_summary} ({numerical_match['overall_percentage']}%) | {' | '.join(parts)}"