#!/usr/bin/env python3
"""
RAG LLM Predictor
Final ICD-10 code prediction using discharge summary + RAG-retrieved codes
"""

import json
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential


class RAGLLMPredictor:
    """Predict ICD-10 codes using LLM with RAG context"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        litellm.set_verbose = False
    
    def _build_candidate_context(self, enriched_codes: Dict) -> str:
        """Build detailed context for candidate codes"""
        
        context_parts = []
        
        for code, data in enriched_codes.items():
            # Basic info
            code_text = f"## Code: {code}\n"
            code_text += f"**Description:** {data['description']}\n"
            code_text += f"**Matched Medical Entities:** {', '.join(data['matched_entities'])}\n"
            code_text += f"**Retrieval Confidence:** {data['avg_score']:.3f}\n"
            
            # Parent
            if data.get('parent'):
                code_text += f"**Parent Code:** {data['parent']}\n"
            
            # Parent chain (hierarchy)
            parent_chain = data.get('parent_chain', [])
            if parent_chain:
                try:
                    # Handle dict format
                    if isinstance(parent_chain[0], dict):
                        chain_str = " → ".join([f"{p['code']} ({p['name']})" for p in parent_chain])
                    # Handle string format
                    else:
                        chain_str = " → ".join(parent_chain)
                    code_text += f"**Hierarchy:** {chain_str}\n"
                except:
                    pass  # Skip if format is unexpected
            
            # Children
            children = data.get('children', [])
            if children:
                try:
                    child_count = len(children)
                    code_text += f"**Has {child_count} child codes:** "
                    # Handle dict format
                    if isinstance(children[0], dict):
                        child_preview = [f"{c['code']}" for c in children[:3]]
                    # Handle string format
                    else:
                        child_preview = children[:3]
                    code_text += ", ".join(child_preview)
                    if child_count > 3:
                        code_text += f" (and {child_count - 3} more)"
                    code_text += "\n"
                except:
                    pass  # Skip if format is unexpected
            
            context_parts.append(code_text)
        
        return "\n".join(context_parts)
    
    def create_prediction_prompt(
        self, 
        discharge_summary: str,
        extracted_entities: List[str],
        enriched_codes: Dict
    ) -> List[Dict[str, str]]:
        """
        Create prompt for final ICD-10 prediction
        """
        
        candidate_context = self._build_candidate_context(enriched_codes)
        
        system_message = """You are an expert medical coding specialist with deep knowledge of ICD-10-CM coding guidelines. 

You will be provided with:
1. A clinical discharge summary
2. Medical entities extracted from that summary
3. Candidate ICD-10 codes retrieved from a knowledge graph based on those entities, with full hierarchical context

Your task is to select the BEST and MOST APPROPRIATE ICD-10 codes that should be assigned to this discharge summary.

Guidelines:
- Only select codes that are clearly supported by the discharge summary
- Follow ICD-10 coding conventions and hierarchy rules
- Use the most specific code when documentation supports it
- Avoid redundancy (don't code both parent and child unless appropriate)
- Consider the entity-to-code matching information provided
- Higher retrieval confidence scores suggest better matches, but always verify against clinical documentation

Provide your response in JSON format with reasoning for each code."""

        user_message = f"""# DISCHARGE SUMMARY

{discharge_summary}

---

# EXTRACTED MEDICAL ENTITIES

The following medical conditions were identified in the discharge summary:
{', '.join([f'"{e}"' for e in extracted_entities])}

---

# CANDIDATE ICD-10 CODES (Retrieved from Knowledge Graph)

You have been provided with {len(enriched_codes)} candidate codes retrieved based on semantic similarity to the extracted entities. Each code includes its full ICD-10 hierarchy.

{candidate_context}

---

# YOUR TASK

Analyze the discharge summary and the candidate codes above. Select the most appropriate ICD-10 codes.

Provide your response in the following JSON format:

```json
{{
  "codes_with_reasoning": [
    {{
      "code": "ICD10_CODE",
      "description": "code description",
      "reasoning": "Why this code is appropriate for this patient",
      "evidence_from_summary": "specific text from discharge summary",
      "matched_entity": "which extracted entity led to this code",
      "decision": "INCLUDE",
      "confidence": "HIGH|MEDIUM|LOW"
    }}
  ],
  "final_codes": ["list", "of", "selected", "ICD10", "codes"],
  "overall_reasoning": "Brief summary of your coding decisions"
}}
```

INSTRUCTIONS:
1. Review each candidate code systematically
2. Verify supporting evidence in the discharge summary
3. Consider hierarchical relationships
4. Select the final code set
5. Provide clear reasoning

Begin your analysis:"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm(self, messages: List[Dict[str, str]]) -> Dict:
        """Call LLM with retry logic"""
        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4000,
                timeout=120
            )
            
            content = response.choices[0].message.content
            
            return {
                'success': True,
                'content': content,
                'model': self.model_name,
                'usage': dict(response.usage) if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def parse_prediction_response(self, response_content: str) -> Dict:
        """Parse LLM prediction response"""
        
        try:
            # Clean up markdown code blocks
            content = response_content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            parsed = json.loads(content)
            
            # Normalize codes (remove dots, uppercase)
            final_codes = parsed.get('final_codes', [])
            normalized_codes = [c.replace('.', '').upper().strip() for c in final_codes]
            
            return {
                'codes_with_reasoning': parsed.get('codes_with_reasoning', []),
                'final_codes': normalized_codes,
                'overall_reasoning': parsed.get('overall_reasoning', ''),
                'response_parsed': True
            }
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse prediction JSON: {str(e)}")
            return {
                'codes_with_reasoning': [],
                'final_codes': [],
                'overall_reasoning': response_content,
                'response_parsed': False,
                'parse_error': str(e)
            }
    
    def predict(
        self,
        discharge_summary: str,
        extracted_entities: List[str],
        enriched_codes: Dict
    ) -> Dict:
        """
        Predict ICD-10 codes using RAG context
        
        Returns:
            {
                'success': bool,
                'predicted_codes': List[str],
                'codes_with_reasoning': List[Dict],
                'overall_reasoning': str,
                'entities_used': List[str],
                'candidates_provided': int,
                'usage': Dict,
                'error': str (if failed)
            }
        """
        
        # Create prompt
        messages = self.create_prediction_prompt(
            discharge_summary,
            extracted_entities,
            enriched_codes
        )
        
        # Call LLM
        response = self.call_llm(messages)
        
        if not response['success']:
            return {
                'success': False,
                'predicted_codes': [],
                'error': response.get('error', 'Unknown error'),
                'entities_used': extracted_entities,
                'candidates_provided': len(enriched_codes)
            }
        
        # Parse response
        parsed = self.parse_prediction_response(response['content'])
        
        return {
            'success': True,
            'predicted_codes': parsed['final_codes'],
            'codes_with_reasoning': parsed['codes_with_reasoning'],
            'overall_reasoning': parsed['overall_reasoning'],
            'response_parsed': parsed.get('response_parsed', True),
            'entities_used': extracted_entities,
            'candidates_provided': len(enriched_codes),
            'usage': response.get('usage', {})
        }


def test_rag_predictor():
    """Test RAG predictor with sample data"""
    
    sample_discharge = """
    DISCHARGE SUMMARY
    
    PATIENT: John Doe, 68 years old
    
    HOSPITAL COURSE:
    The patient presented with chest pain and was diagnosed with ST-elevation myocardial infarction.
    He underwent cardiac catheterization with stent placement. During hospitalization, developed
    atrial fibrillation with rapid ventricular response, which was rate-controlled. Also noted to
    have congestive heart failure, managed with diuresis. Past medical history significant for
    diabetes mellitus type 2, hypertension, and hyperlipidemia.
    
    DISCHARGE DIAGNOSES:
    1. ST-elevation myocardial infarction
    2. Atrial fibrillation
    3. Congestive heart failure
    4. Diabetes mellitus type 2
    5. Hypertension
    """
    
    sample_entities = [
        "ST-elevation myocardial infarction",
        "atrial fibrillation",
        "congestive heart failure",
        "diabetes mellitus type 2",
        "hypertension"
    ]
    
    sample_enriched_codes = {
        "I2101": {
            "code": "I2101",
            "description": "ST elevation (STEMI) myocardial infarction involving left main coronary artery",
            "matched_entities": ["ST-elevation myocardial infarction"],
            "avg_score": 0.92,
            "parent": "I21",
            "parent_chain": [{"code": "I20-I25", "name": "Ischemic heart diseases"}],
            "children": []
        },
        "I4891": {
            "code": "I4891",
            "description": "Unspecified atrial fibrillation",
            "matched_entities": ["atrial fibrillation"],
            "avg_score": 0.95,
            "parent": "I48",
            "parent_chain": [{"code": "I30-I52", "name": "Other forms of heart disease"}],
            "children": []
        },
        "I5023": {
            "code": "I5023",
            "description": "Acute on chronic systolic (congestive) heart failure",
            "matched_entities": ["congestive heart failure"],
            "avg_score": 0.88,
            "parent": "I50",
            "parent_chain": [{"code": "I30-I52", "name": "Other forms of heart disease"}],
            "children": []
        }
    }
    
    print("="*80)
    print("TESTING RAG LLM PREDICTOR")
    print("="*80)
    print()
    
    predictor = RAGLLMPredictor(model_name="gpt-4o")
    
    print("Making prediction...")
    result = predictor.predict(
        discharge_summary=sample_discharge,
        extracted_entities=sample_entities,
        enriched_codes=sample_enriched_codes
    )
    
    if result['success']:
        print(f"\n✅ Prediction successful!")
        print(f"\nPredicted codes ({len(result['predicted_codes'])}):")
        for code in result['predicted_codes']:
            print(f"  - {code}")
        
        print(f"\nEntities used: {len(result['entities_used'])}")
        print(f"Candidates provided: {result['candidates_provided']}")
        
        if result.get('usage'):
            print(f"\nToken usage: {result['usage']}")
    else:
        print(f"\n❌ Prediction failed: {result.get('error')}")
    
    print()
    print("="*80)


if __name__ == '__main__':
    test_rag_predictor()
