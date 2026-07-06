"""
llm_integration.py
===================
Thin, provider-agnostic wrapper around the LLM / GenAI call so the rest of the
codebase (pipeline.py, app.py) never needs to know which backend is in use.

Supported providers (set LLM_PROVIDER in config.py / .env):
    - "openai"      : OpenAI GPT models (gpt-4o-mini, gpt-4o, ...)
    - "gemini"      : Google Gemini via google-generativeai
    - "mistral"     : Mistral API
    - "llama_local" : Local Llama 3 via Hugging Face Transformers (no API key needed)

Only the selected provider's SDK needs to be installed/configured at runtime.
"""

from pathlib import Path
from typing import List, Dict

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class DummyLLM:
    def __init__(self, provider: str):
        self.provider = provider

    def chat(self, messages: List[Dict], temperature: float = None, max_tokens: int = None) -> str:
        messages = messages or []
        if not isinstance(messages, list):
            try:
                messages = list(messages)
            except Exception:
                messages = []

        user_message = ""
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                user_message = message.get("content", "")
                break

        if not isinstance(user_message, str) or not user_message.strip():
            return "LLM is not configured. Please set LLM_API_KEY in .env or change LLM_PROVIDER."

        return (
            "🚧 LLM not configured: the app is running in fallback mode. "
            "Set LLM_API_KEY in .env for OpenAI or configure a supported provider to enable chat/report generation."
        )


class LLMClient:
    def __init__(self, provider: str = None, model_name: str = None, api_key: str = None):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.model_name = model_name or config.LLM_MODEL_NAME
        self.api_key = api_key or config.LLM_API_KEY
        self._client = self._init_client()

    # ------------------------------------------------------------------
    def _init_client(self):
        try:
            if self.provider == "openai":
                if not self.api_key:
                    print("[WARN] LLM_API_KEY is missing. Falling back to dummy LLM client.")
                    return DummyLLM(self.provider)
                from openai import OpenAI
                return OpenAI(api_key=self.api_key)

            elif self.provider == "gemini":
                if not self.api_key:
                    print("[WARN] Gemini API key is missing. Falling back to dummy LLM client.")
                    return DummyLLM(self.provider)
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                return genai.GenerativeModel(self.model_name)

            elif self.provider == "mistral":
                if not self.api_key:
                    print("[WARN] Mistral API key is missing. Falling back to dummy LLM client.")
                    return DummyLLM(self.provider)
                from mistralai import Mistral
                return Mistral(api_key=self.api_key)

            elif self.provider == "llama_local":
                try:
                    from transformers import pipeline as hf_pipeline
                except ImportError:
                    print("[WARN] Transformers is not installed. Falling back to dummy LLM client.")
                    return DummyLLM(self.provider)
                return hf_pipeline(
                    "text-generation",
                    model=self.model_name or "meta-llama/Meta-Llama-3-8B-Instruct",
                    device_map="auto",
                )

            else:
                print(f"[WARN] Unsupported LLM_PROVIDER '{self.provider}'. Falling back to dummy LLM client.")
                return DummyLLM(self.provider)
        except Exception as exc:
            print(f"[WARN] Failed to initialize LLM provider '{self.provider}': {exc}")
            return DummyLLM(self.provider)

    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict], temperature: float = None, max_tokens: int = None) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        Returns the assistant's reply text.
        """
        temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        if isinstance(self._client, DummyLLM):
            return self._client.chat(messages, temperature=temperature, max_tokens=max_tokens)

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "gemini":
            # Gemini uses a single running chat session; flatten history into one prompt.
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            response = self._client.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return response.text.strip()

        elif self.provider == "mistral":
            response = self._client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        elif self.provider == "llama_local":
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            output = self._client(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
            )
            generated = output[0]["generated_text"]
            return generated[len(prompt):].strip()

        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {self.provider}")


if __name__ == "__main__":
    client = LLMClient()
    demo_messages = [
        {"role": "system", "content": "You are a helpful retail assistant."},
        {"role": "user", "content": "Say hello in one sentence."},
    ]
    print(client.chat(demo_messages))
