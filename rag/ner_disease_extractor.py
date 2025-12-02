#!/usr/bin/env python3
"""
NER Disease Extractor
Extracts medical conditions, diseases, and symptoms from discharge summaries using GPT-4
"""

import json
import os
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential


class NERDiseaseExtractor:
    """Extract medical entities from clinical text using LLM"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        litellm.set_verbose = False
    
    def create_ner_prompt(self, discharge_summary: str) -> List[Dict[str, str]]:
        """
        Create prompt for NER extraction
        """
        system_message = """You are an expert medical NER (Named Entity Recognition) system specialized in extracting medical conditions from clinical discharge summaries.

Your task is to extract ALL medical conditions, diseases, symptoms, and diagnoses mentioned in the discharge summary.

IMPORTANT:
- Extract the exact medical terms as they appear in the text
- Include primary diagnoses, secondary diagnoses, complications, and comorbidities
- Include both acute and chronic conditions
- Include symptoms if they represent distinct clinical findings
- DO NOT include procedures, medications, or lab test names
- Use standard medical terminology

Return your response as a JSON object with this structure:
{
  "entities": [
    {
      "entity": "exact medical term",
      "category": "diagnosis|symptom|condition",
      "context": "brief context from text"
    }
  ]
}"""

        user_message = f"""# DISCHARGE SUMMARY

{discharge_summary}

---

Extract all medical conditions, diseases, and symptoms from the above discharge summary. Return as JSON."""

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
                temperature=0.0,  # Deterministic for extraction
                max_tokens=2000,
                timeout=60
            )
            
            content = response.choices[0].message.content
            
            return {
                'success': True,
                'content': content,
                'usage': dict(response.usage) if hasattr(response, 'usage') else {}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def parse_ner_response(self, response_content: str) -> List[str]:
        """
        Parse NER response and extract disease names
        """
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
            
            # Extract entity names
            entities = parsed.get('entities', [])
            disease_names = []
            
            for entity in entities:
                entity_name = entity.get('entity', '').strip()
                if entity_name:
                    disease_names.append(entity_name)
            
            # Deduplicate while preserving order
            seen = set()
            unique_names = []
            for name in disease_names:
                name_lower = name.lower()
                if name_lower not in seen:
                    seen.add(name_lower)
                    unique_names.append(name)
            
            return unique_names
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse NER JSON response: {str(e)}")
            return []
    
    def extract_diseases(self, discharge_summary: str) -> Dict:
        """
        Extract disease entities from discharge summary
        
        Returns:
            {
                'success': bool,
                'entities': List[str],
                'raw_response': str,
                'error': str (if failed)
            }
        """
        # Create prompt
        messages = self.create_ner_prompt(discharge_summary)
        
        # Call LLM
        response = self.call_llm(messages)
        
        if not response['success']:
            return {
                'success': False,
                'entities': [],
                'error': response.get('error', 'Unknown error'),
                'raw_response': None
            }
        
        # Parse response
        entities = self.parse_ner_response(response['content'])
        
        return {
            'success': True,
            'entities': entities,
            'raw_response': response['content'],
            'usage': response.get('usage', {})
        }


def test_ner_extractor():
    """Test the NER extractor on a sample discharge summary"""
    
    sample_discharge = """
    DISCHARGE SUMMARY
    
    PATIENT: John Doe
    AGE: 68 years
    
    ADMISSION DIAGNOSIS:
    1. Acute myocardial infarction
    2. Atrial fibrillation with rapid ventricular response
    
    HOSPITAL COURSE:
    The patient is a 68-year-old male with a history of hypertension, diabetes mellitus type 2, 
    and hyperlipidemia who presented with chest pain and was found to have ST-elevation 
    myocardial infarction. He underwent emergent cardiac catheterization with placement of 
    drug-eluting stent to the left anterior descending artery.
    
    During hospitalization, the patient developed atrial fibrillation with rapid ventricular 
    response which was managed with rate control medications. He also had episodes of 
    congestive heart failure which improved with diuresis.
    
    DISCHARGE DIAGNOSES:
    1. ST-elevation myocardial infarction
    2. Atrial fibrillation
    3. Congestive heart failure
    4. Diabetes mellitus type 2
    5. Hypertension
    6. Hyperlipidemia
    """
    
    print("="*80)
    print("TESTING NER DISEASE EXTRACTOR")
    print("="*80)
    print()
    
    extractor = NERDiseaseExtractor(model_name="gpt-4o")
    
    print("Sample discharge summary:")
    print("-"*80)
    print(sample_discharge[:300] + "...")
    print()
    
    print("Extracting medical entities...")
    result = extractor.extract_diseases(sample_discharge)
    
    if result['success']:
        print(f"\n✅ Successfully extracted {len(result['entities'])} entities:")
        print()
        for i, entity in enumerate(result['entities'], 1):
            print(f"  {i}. {entity}")
        print()
        
        if result.get('usage'):
            print(f"Token usage: {result['usage']}")
    else:
        print(f"\n❌ Failed to extract entities: {result.get('error')}")
    
    print()
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract medical entities from discharge summary')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample data')
    parser.add_argument('--text', type=str,
                       help='Discharge summary text to process')
    parser.add_argument('--model', default='gpt-4o',
                       help='LLM model to use')
    
    args = parser.parse_args()
    
    if args.test:
        test_ner_extractor()
    elif args.text:
        extractor = NERDiseaseExtractor(model_name=args.model)
        result = extractor.extract_diseases(args.text)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()
