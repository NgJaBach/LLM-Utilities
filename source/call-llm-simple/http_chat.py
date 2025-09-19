import requests
import json
import backoff
import re
from typing import Dict, Any, Optional
from ..constants import BAILAB_HTTP

class OllamaError(Exception):
    """Custom exception for Ollama API errors"""
    pass

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, OllamaError),
    max_time=999, max_tries=9999
)
def ollama_completion_with_backoff(**kwargs) -> Dict[str, Any]:
    """Make a request to Ollama API with exponential backoff"""
    try:
        response = requests.post(BAILAB_HTTP, json=kwargs)
        response.raise_for_status()
        return json.loads(response.text)
    except requests.exceptions.HTTPError as e:
        raise OllamaError(f"HTTP error: {e}")
    except json.JSONDecodeError:
        raise OllamaError("Invalid JSON response")

def remove_reasoning(response_content: str) -> str:
    """Remove reasoning part if present"""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return response_content.strip()

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "deepseek-v2:16b",
    max_tokens: int = 128000,
    temperature: float = 0.3,
    reasoning_level: Optional[str] = None,
) -> str:
    """Generate text using Ollama API with similar interface to OpenAI"""
    # Combine system and user prompts if system prompt is provided
    prompt = f"{sys_prompt}\n\n{user_prompt}" if sys_prompt else user_prompt
    
    # Prepare request parameters
    params = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    # Add reasoning parameters if specified
    if reasoning_level:
        params["options"]["reasoning_effort"] = reasoning_level
    
    # Make the API call
    response = ollama_completion_with_backoff(**params)
    
    # Extract and clean response
    result = response.get("response", "")
    return remove_reasoning(result)

if __name__ == "__main__":
    # Example usage
    response = ask("What is the capital of France?")
    print("Response:", response)
    
    # Example with system prompt and reasoning
    sys_prompt = "You are a helpful assistant."
    response = ask("Explain the theory of relativity.", sys_prompt=sys_prompt, reasoning_level="high")
    print("Response with reasoning:", response)