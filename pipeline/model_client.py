"""
Model client implementations.
Provides a unified interface for calling various model APIs.
"""
import base64
import io
from typing import Optional, Dict, Any
from PIL import Image
import requests


class BaseModelClient:
    """Base model client."""

    def generate(self, prompt: str, image: Optional[Image.Image] = None,
                 max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseModelClient):
    """OpenAI-compatible API client."""

    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def generate(self, prompt: str, image: Optional[Image.Image] = None,
                 max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []

        if image:
            image_b64 = self._image_to_base64(image)
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        else:
            content = prompt

        messages.append({
            "role": "user",
            "content": content
        })

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        response = requests.post(url, json=data, headers=headers, timeout=300)
        response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected API response: {result}")


class VLLMClient(BaseModelClient):
    """vLLM server client."""

    def __init__(self, api_base: str, model: str, api_key: Optional[str] = None):
        self.api_base = api_base.rstrip('/')
        self.model = model
        self.api_key = api_key

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def generate(self, prompt: str, image: Optional[Image.Image] = None,
                 max_tokens: int = 2048, temperature: float = 0.7, **kwargs) -> str:
        url = f"{self.api_base}/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []

        if image:
            image_b64 = self._image_to_base64(image)
            content = [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }
                }
            ]
        else:
            content = prompt

        messages.append({
            "role": "user",
            "content": content
        })

        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        response = requests.post(url, json=data, headers=headers, timeout=300)
        response.raise_for_status()

        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unexpected API response: {result}")


def create_model_client(config: Dict[str, Any]) -> BaseModelClient:
    """Create a model client from a config dict."""
    client_type = config.get('type', 'openai').lower()
    api_base = config['api_base']
    api_key = config.get('api_key', '')
    model = config['model']

    if client_type in ['openai', 'openai_compatible']:
        return OpenAICompatibleClient(api_base, api_key, model)
    elif client_type == 'vllm':
        return VLLMClient(api_base, model, api_key)
    else:
        raise ValueError(f"Unknown client type: {client_type}")
