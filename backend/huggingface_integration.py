"""
HuggingFace Integration - Model discovery and validation utilities
"""
import logging
import requests
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)

class HuggingFaceModelManager:
    """Manages HuggingFace model discovery and validation"""
    
    def __init__(self):
        self.model_cache = {}
        self.hf_api_base = "https://huggingface.co/api"
        
    def validate_model(self, model_name: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Validate if a HuggingFace model exists and get its information
        
        Returns:
            - is_valid: bool indicating if model exists
            - message: status message
            - model_info: dict with model information if valid
        """
        try:
            # Check cache first
            if model_name in self.model_cache:
                return True, "Model validated (cached)", self.model_cache[model_name]
            
            # Fetch model information via API
            response = requests.get(f"{self.hf_api_base}/models/{model_name}", timeout=10)
            
            if response.status_code == 200:
                info = response.json()
                
                # Extract relevant information
                model_data = {
                    'name': model_name,
                    'downloads': info.get('downloads', 0),
                    'likes': info.get('likes', 0),
                    'library_name': info.get('library_name', 'unknown'),
                    'pipeline_tag': info.get('pipeline_tag', 'unknown'),
                    'tags': info.get('tags', []),
                    'model_type': self._extract_model_type_from_dict(info),
                    'size_estimate': self._estimate_model_size_from_name(model_name),
                    'is_gated': info.get('gated', False),
                    'license': info.get('cardData', {}).get('license', 'unknown')
                }
                
                # Cache the result
                self.model_cache[model_name] = model_data
                
                return True, "Model validated successfully", model_data
            
            elif response.status_code == 404:
                return False, "Model not found on HuggingFace Hub", None
            elif response.status_code == 401 or response.status_code == 403:
                return False, "Model is private or requires authentication", None
            else:
                return False, f"Error accessing model (HTTP {response.status_code})", None
                
        except requests.RequestException as e:
            logger.error(f"Network error validating model {model_name}: {e}")
            return False, f"Network error: {str(e)}", None
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def _extract_model_type_from_dict(self, info: dict) -> str:
        """Extract model type from model info dict"""
        if 'config' in info and info['config']:
            return info['config'].get('model_type', 'unknown')
        
        # Try to infer from tags
        tags = info.get('tags', [])
        tags_str = ' '.join(tags).lower()
        if 'llama' in tags_str:
            return 'llama'
        elif 'mistral' in tags_str:
            return 'mistral'
        elif 'gpt' in tags_str:
            return 'gpt'
        elif 'bloom' in tags_str:
            return 'bloom'
        elif 'falcon' in tags_str:
            return 'falcon'
        
        return 'unknown'
    
    def _estimate_model_size_from_name(self, model_name: str) -> str:
        """Estimate model size from model name"""
        model_name = model_name.lower()
        
        size_indicators = {
            '7b': '7B parameters (~13GB)',
            '13b': '13B parameters (~26GB)', 
            '30b': '30B parameters (~60GB)',
            '65b': '65B parameters (~130GB)',
            '70b': '70B parameters (~140GB)',
            '8x7b': '8x7B parameters (~90GB)',
            '3b': '3B parameters (~6GB)',
            '1.5b': '1.5B parameters (~3GB)',
            '6b': '6B parameters (~12GB)',
            '12b': '12B parameters (~24GB)'
        }
        
        for size_key, size_desc in size_indicators.items():
            if size_key in model_name:
                return size_desc
        
        return 'Unknown size'
    
    def search_models(self, query: str, limit: int = 20, token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for models on HuggingFace Hub"""
        try:
            # Use HuggingFace API to search models
            params = {
                'search': query,
                'filter': 'text-generation',
                'sort': 'downloads',
                'limit': limit
            }
            
            headers = {}
            if token:
                headers['Authorization'] = f'Bearer {token}'
            
            response = requests.get(f"{self.hf_api_base}/models", params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                models = response.json()
                
                results = []
                for model in models:
                    results.append({
                        'name': model.get('id', ''),
                        'downloads': model.get('downloads', 0),
                        'likes': model.get('likes', 0),
                        'library_name': model.get('library_name', 'transformers'),
                        'tags': model.get('tags', [])
                    })
                
                return results
            else:
                logger.error(f"Search API returned {response.status_code}")
                return []
            
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []
    
    def get_popular_llm_models(self) -> List[str]:
        """Get a list of popular LLM models"""
        popular_models = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-v0.1",
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "codellama/CodeLlama-7b-Python-hf",
            "codellama/CodeLlama-7b-Instruct-hf",
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "huggingfaceh4/zephyr-7b-beta",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "stabilityai/stablelm-3b-4e1t",
            "EleutherAI/gpt-j-6b",
            "bigscience/bloom-7b1",
            "tiiuae/falcon-7b",
            "tiiuae/falcon-7b-instruct"
        ]
        return popular_models
    
    def is_model_suitable_for_fine_tuning(self, model_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a model is suitable for fine-tuning"""
        if not model_info:
            return False, "No model information available"
        
        # Check if it's a text generation model
        pipeline_tag = model_info.get('pipeline_tag', '')
        if pipeline_tag != 'text-generation' and pipeline_tag != 'unknown':
            return False, f"Model is designed for '{pipeline_tag}', not text generation"
        
        # Check if it's gated (requires approval)
        if model_info.get('is_gated', False):
            return False, "Model requires approval/authentication to access"
        
        # Check library compatibility
        library_name = model_info.get('library_name', '')
        if library_name not in ['transformers', 'unknown']:
            return False, f"Model uses '{library_name}' library, transformers required"
        
        # Size warnings
        size_estimate = model_info.get('size_estimate', '')
        if '70b' in size_estimate.lower() or '65b' in size_estimate.lower():
            return True, "⚠️ Warning: Very large model, may require significant resources"
        elif '30b' in size_estimate.lower() or '13b' in size_estimate.lower():
            return True, "⚠️ Warning: Large model, ensure adequate GPU memory"
        
        return True, "Model appears suitable for fine-tuning"

# Global instance
hf_manager = HuggingFaceModelManager()

def get_hf_manager() -> HuggingFaceModelManager:
    """Get the global HuggingFace manager instance"""
    return hf_manager