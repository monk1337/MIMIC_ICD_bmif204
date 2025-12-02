#!/usr/bin/env python3
"""
LLM API Caller - Step 1 of Evaluation
Calls LLM API and saves raw responses to disk
"""

import os
import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging
from tqdm import tqdm
import time
import traceback
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LiteLLM imports
from litellm import completion
import litellm


class LLMAPICaller:
    """Call LLM APIs and save raw responses"""
    
    def __init__(self, config_path: str, model_name: str):
        """Initialize API caller"""
        self.config = self.load_config(config_path)
        self.model_name = model_name
        self.model_config = self.get_model_config(model_name)
        
        self.setup_logging()
        self.setup_output_dir()
        
        # Disable litellm verbose logging by default
        litellm.set_verbose = False
        
        logging.info("="*80)
        logging.info(f"LLM API CALLER - Model: {model_name}")
        logging.info("="*80)
    
    def load_config(self, config_path: str) -> Dict:
        """Load YAML configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for specific model"""
        for model in self.config['models']:
            if model['name'] == model_name:
                return model
        
        raise ValueError(f"Model '{model_name}' not found in config. Available models: {[m['name'] for m in self.config['models']]}")
    
    def setup_logging(self):
        """Setup logging"""
        log_level = getattr(logging, self.config['evaluation']['log_level'])
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    
    def setup_output_dir(self):
        """Create output directory for this model"""
        # Convert model path to safe folder name (replace / with _)
        safe_name = self.model_config['model'].replace('/', '_')
        
        # Create base directory
        base_dir = Path(self.config['evaluation']['output_dir'])
        base_dir.mkdir(exist_ok=True)
        
        # Create model-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = base_dir / f"{safe_name}_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Save model config
        config_file = self.output_dir / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        logging.info(f"Output directory: {self.output_dir}")
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load evaluation sample"""
        sample_file = self.config['evaluation']['sample_file']
        logging.info(f"Loading sample data: {sample_file}")
        
        df = pd.read_csv(sample_file)
        logging.info(f"Loaded {len(df)} samples")
        
        # Parse labels
        df['actual_codes'] = df['labels'].apply(lambda x: set(x.split(';')))
        
        return df
    
    def create_prompt(self, text: str) -> List[Dict[str, str]]:
        """Create prompt messages for LLM"""
        system_msg = self.config['prompt']['system_message']
        user_template = self.config['prompt']['user_message_template']
        
        user_msg = user_template.format(text=text)
        
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    
    def call_llm(self, messages: List[Dict]) -> Dict:
        """Call LLM with retry logic"""
        max_retries = self.config['evaluation']['max_retries']
        retry_delay = self.config['evaluation']['retry_delay']
        timeout = self.config['evaluation']['request_timeout']
        
        # Set API key from environment
        api_key_env = self.model_config.get('api_key_env')
        if api_key_env and api_key_env in os.environ:
            os.environ[api_key_env.split('/')[-1]] = os.environ[api_key_env]
        
        # Set custom API base if provided
        api_base = self.model_config.get('api_base')
        
        for attempt in range(max_retries):
            try:
                # Build completion params
                model_name = self.model_config['model'].lower()
                
                # GPT-5 and o-series models require temperature=1.0 (fixed)
                if 'gpt-5' in model_name or 'o3' in model_name or 'o4' in model_name or 'o1' in model_name:
                    temperature = 1.0
                else:
                    temperature = self.model_config.get('temperature', 0.7)
                
                completion_params = {
                    'model': self.model_config['model'],
                    'messages': messages,
                    'temperature': temperature,
                    'timeout': timeout
                }
                
                # Add custom API base and key if provided (for custom endpoints)
                if api_base:
                    completion_params['api_base'] = api_base
                    # When using custom api_base, explicitly pass the API key
                    if api_key_env and api_key_env in os.environ:
                        completion_params['api_key'] = os.environ[api_key_env]
                
                # GPT-5 and o-series models use max_completion_tokens instead of max_tokens
                if 'gpt-5' in model_name or 'o3' in model_name or 'o4' in model_name or 'o1' in model_name:
                    completion_params['max_completion_tokens'] = self.model_config.get('max_tokens', 2000)
                else:
                    completion_params['max_tokens'] = self.model_config.get('max_tokens', 2000)
                
                # Add response format for non-reasoning models
                if 'reasoning_effort' not in self.model_config and 'reasoner' not in self.model_config['model'].lower():
                    completion_params['response_format'] = {
                        "type": self.config['prompt'].get('response_format', 'text')
                    }
                
                # Handle reasoning effort - Anthropic uses 'thinking', others use 'reasoning_effort'
                if 'reasoning_effort' in self.model_config:
                    if 'anthropic' in model_name or 'claude' in model_name:
                        # Anthropic uses 'thinking' parameter and requires temperature=1.0
                        # Minimum budget_tokens is 1024
                        budget_map = {'low': 1024, 'medium': 2048, 'high': 4096}
                        effort = self.model_config['reasoning_effort']
                        completion_params['thinking'] = {
                            'type': 'enabled',
                            'budget_tokens': budget_map.get(effort, 2000)
                        }
                        # Anthropic requires temperature=1.0 when thinking is enabled
                        completion_params['temperature'] = 1.0
                    else:
                        # Other models use 'reasoning_effort'
                        completion_params['reasoning_effort'] = self.model_config['reasoning_effort']
                
                start_time = time.time()
                response = completion(**completion_params)
                end_time = time.time()
                
                # Extract all relevant data
                result = {
                    'success': True,
                    'model': response.model,
                    'content': response.choices[0].message.content,
                    'reasoning_content': getattr(response.choices[0].message, 'reasoning_content', None),
                    'finish_reason': response.choices[0].finish_reason,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'response_time': end_time - start_time,
                    'timestamp': datetime.now().isoformat()
                }
                
                return result
                
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return {
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.now().isoformat()
                    }
    
    def process_sample(self, row: pd.Series) -> Dict:
        """Process a single sample"""
        sample_id = row['sample_id']
        
        # Create prompt
        messages = self.create_prompt(row['text'])
        
        # Call LLM
        llm_result = self.call_llm(messages)
        
        # Build output with metadata
        result = {
            'sample_id': int(sample_id),
            'hadm_id': int(row['hadm_id']),
            'subject_id': int(row['subject_id']),
            'text': row['text'],
            'actual_codes': sorted(row['actual_codes']),
            'num_actual_codes': len(row['actual_codes']),
            'stratification_group': row['stratification_group'],
            'code_frequency_tier': row['code_frequency_tier'],
            'length_tier': row['length_tier'],
            'text_length': int(row['length']),
            'messages': messages,
            'llm_response': llm_result
        }
        
        return result
    
    def run(self):
        """Run API calls for all samples"""
        df = self.load_sample_data()
        
        logging.info(f"\nProcessing {len(df)} samples with model: {self.model_name}")
        logging.info(f"Model: {self.model_config['model']}")
        logging.info("\n" + "="*80 + "\n")
        
        results = []
        failed_count = 0
        
        # Process with progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="API Calls"):
            result = self.process_sample(row)
            results.append(result)
            
            if not result['llm_response']['success']:
                failed_count += 1
            
            # Save intermediate results every 10 samples
            if (idx + 1) % 10 == 0:
                self.save_results(results, 'intermediate')
        
        # Save final results
        self.save_results(results, 'final')
        
        # Print summary
        logging.info("\n" + "="*80)
        logging.info("API CALLS COMPLETE!")
        logging.info("="*80)
        logging.info(f"Total Samples: {len(results)}")
        logging.info(f"Successful: {len(results) - failed_count}")
        logging.info(f"Failed: {failed_count}")
        logging.info(f"Output directory: {self.output_dir}")
        logging.info("="*80 + "\n")
        
        return self.output_dir
    
    def save_results(self, results: List[Dict], suffix: str = 'final'):
        """Save results to JSON file"""
        output_file = self.output_dir / f"responses_{suffix}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Also save metadata
        metadata = {
            'model_name': self.model_name,
            'model_path': self.model_config['model'],
            'total_samples': len(results),
            'successful': len([r for r in results if r['llm_response']['success']]),
            'failed': len([r for r in results if not r['llm_response']['success']]),
            'timestamp': datetime.now().isoformat(),
            'config': self.model_config
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Call LLM API and save responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 llm_call_api.py --model gpt-4o
  python3 llm_call_api.py --model claude-3.5-sonnet
  python3 llm_call_api.py --model qwen-235b --config my_config.yaml
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                        help="Model name from config (e.g., gpt-4o, claude-3.5-sonnet)")
    parser.add_argument("--config", type=str, default="llm_eval_config.yaml",
                        help="Path to config YAML file")
    
    args = parser.parse_args()
    
    try:
        # Initialize caller
        caller = LLMAPICaller(
            config_path=args.config,
            model_name=args.model
        )
        
        # Run API calls
        output_dir = caller.run()
        
        print(f"\n✅ API calls complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"\n▶️  Next step: Run evaluation")
        print(f"   python3 llm_evaluate_results.py --folder {output_dir.name}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logging.error(f"Fatal error: {str(e)}")
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
