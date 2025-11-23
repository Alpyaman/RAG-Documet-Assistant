"""
LLM Generator - Handles answer generation from retrieved context.

Provides an abstract interface for multiple LLM providers:
- OpenAI (GPT-3.5/GPT-4)
- Ollama (Local models)
- HuggingFace (Open source models)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""

    answer: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    context_used: Optional[List[str]] = None

    def __repr__(self) -> str:
        return (
            f"GenerationResult(\n"
            f"  answer_length={len(self.answer)} chars,\n"
            f"  model={self.model},\n"
            f"  total_tokens={self.total_tokens}\n"
            f")"
        )


class BaseLLMGenerator(ABC):
    """
    Abstract base class for LLM generators.

    This allows swapping between different LLM providers
    (OpenAI, Ollama, HuggingFace) without changing the pipeline code.
    """

    def __init__(
        self, model_name: str, temperature: float = 0.7, max_tokens: int = 512
    ):
        """
        Initialize the LLM generator.

        Args:
            model_name: Name of the model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    @abstractmethod
    def generate(self, query: str, context: List[str]) -> GenerationResult:
        """
        Generate an answer based on query and context.

        Args:
            query: User's question
            context: List of relevant text chunks

        Returns:
            GenerationResult with the generated answer
        """
        pass

    def build_prompt(self, query: str, context: List[str]) -> str:
        """
        Build a prompt from query and context chunks.

        This can be overridden by subclasses for custom prompting.

        Args:
            query: User's question
            context: List of relevant text chunks

        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join(
            [f"[{i+1}] {chunk}" for i, chunk in enumerate(context)]
        )

        prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context_text}

Question: {query}

Instructions:
- Answer the question using ONLY the information from the context above
- If the context doesn't contain enough information, say so clearly
- Be concise but complete
- Cite which context section(s) you used (e.g., [1], [2])

Answer:"""

        return prompt

    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        return {
            "provider": self.__class__.__name__,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class OpenAIGenerator(BaseLLMGenerator):
    """
    OpenAI-based LLM generator using GPT models.

    Requires OPENAI_API_KEY environment variable.

    Example:
        >>> generator = OpenAIGenerator(model_name="gpt-3.5-turbo")
        >>> result = generator.generate("What is AI?", ["AI is artificial intelligence..."])
        >>> print(result.answer)
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        """
        Initialize OpenAI generator.

        Args:
            model_name: OpenAI model (e.g., gpt-3.5-turbo, gpt-4)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: OpenAI API key (uses env var if not provided)
        """
        super().__init__(model_name, temperature, max_tokens)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        import os

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"OpenAI client initialized with model: {model_name}")

    def generate(self, query: str, context: List[str]) -> GenerationResult:
        """
        Generate answer using OpenAI API.

        Args:
            query: User's question
            context: List of relevant text chunks

        Returns:
            GenerationResult with the generated answer
        """
        prompt = self.build_prompt(query, context)

        try:
            logger.info(f"Generating answer with {self.model_name}...")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            result = GenerationResult(
                answer=answer,
                model=self.model_name,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                context_used=context,
            )

            logger.info(
                f"Generated answer: {len(answer)} chars, {result.total_tokens} tokens"
            )
            return result

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise


class OllamaGenerator(BaseLLMGenerator):
    """
    Ollama-based local LLM generator.

    Requires Ollama to be installed and running locally.
    https://ollama.ai

    Example:
        >>> generator = OllamaGenerator(model_name="llama2")
        >>> result = generator.generate("What is AI?", ["AI is artificial intelligence..."])
        >>> print(result.answer)
    """

    def __init__(
        self,
        model_name: str = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 512,
        base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Ollama generator.

        Args:
            model_name: Ollama model name (e.g., llama2, mistral, codellama)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            base_url: Ollama server URL
        """
        super().__init__(model_name, temperature, max_tokens)

        self.base_url = base_url

        try:
            import requests

            self.requests = requests
        except ImportError:
            raise ImportError(
                "requests package not installed. Install with: pip install requests"
            )

        # Test connection
        try:
            response = self.requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Ollama server not responding at {base_url}")
            logger.info(f"Connected to Ollama server at {base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Make sure Ollama is installed and running. Error: {e}"
            )

    def generate(self, query: str, context: List[str]) -> GenerationResult:
        """
        Generate answer using Ollama.

        Args:
            query: User's question
            context: List of relevant text chunks

        Returns:
            GenerationResult with the generated answer
        """
        prompt = self.build_prompt(query, context)

        try:
            logger.info(f"Generating answer with Ollama ({self.model_name})...")

            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120,  # Ollama can be slow on first run
            )

            response.raise_for_status()
            result_data = response.json()

            answer = result_data.get("response", "")

            result = GenerationResult(
                answer=answer,
                model=self.model_name,
                prompt_tokens=result_data.get("prompt_eval_count"),
                completion_tokens=result_data.get("eval_count"),
                total_tokens=(
                    result_data.get("prompt_eval_count", 0)
                    + result_data.get("eval_count", 0)
                ),
                context_used=context,
            )

            logger.info(f"Generated answer: {len(answer)} chars")
            return result

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class HuggingFaceGenerator(BaseLLMGenerator):
    """
    HuggingFace-based LLM generator using transformers.

    Runs models locally using the transformers library.
    Good for privacy and cost optimization.

    Example:
        >>> generator = HuggingFaceGenerator(model_name="google/flan-t5-base")
        >>> result = generator.generate("What is AI?", ["AI is artificial intelligence..."])
        >>> print(result.answer)
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: str = "cpu",
    ):
        """
        Initialize HuggingFace generator.

        Args:
            model_name: HuggingFace model name
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__(model_name, temperature, max_tokens)

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: pip install transformers torch"
            )

        self.device = device
        self.torch = torch

        logger.info(f"Loading model {model_name} on {device}...")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Try Seq2Seq first (for models like T5, BART)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model_type = "seq2seq"
        except Exception as e:
            # Fall back to CausalLM (for models like GPT-2, Llama)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model_type = "causal"
            logger.warning(f"Fallback to CausalLM: {e}")

        self.model.to(device)
        self.model.eval()

        logger.info(f"Model loaded successfully (type: {self.model_type})")

    def generate(self, query: str, context: List[str]) -> GenerationResult:
        """
        Generate answer using HuggingFace model.

        Args:
            query: User's question
            context: List of relevant text chunks

        Returns:
            GenerationResult with the generated answer
        """
        prompt = self.build_prompt(query, context)

        try:
            logger.info(f"Generating answer with {self.model_name}...")

            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            # Generate
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                )

            # Decode output
            if self.model_type == "seq2seq":
                answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                # For causal models, skip the input tokens
                answer = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
            # Estimate token counts
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = len(self.tokenizer.encode(answer))

            result = GenerationResult(
                answer=answer.strip(),
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                context_used=context,
            )

            logger.info(f"Generated answer: {len(answer)} chars")
            return result

        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
            raise


def create_generator(
    provider: str = "openai", model_name: Optional[str] = None, **kwargs
) -> BaseLLMGenerator:
    """
    Factory function to create an LLM generator.

    Args:
        provider: Provider name ('openai', 'ollama', 'huggingface')
        model_name: Model name (uses provider defaults if not specified)
        **kwargs: Additional arguments for the generator

    Returns:
        Initialized LLM generator

    Example:
        >>> # OpenAI
        >>> generator = create_generator("openai", model_name="gpt-4")

        >>> # Ollama (local)
        >>> generator = create_generator("ollama", model_name="llama2")

        >>> # HuggingFace (local)
        >>> generator = create_generator("huggingface", model_name="google/flan-t5-base")
    """
    provider = provider.lower()

    if provider == "openai":
        model_name = model_name or "gpt-3.5-turbo"
        return OpenAIGenerator(model_name=model_name, **kwargs)

    elif provider == "ollama":
        model_name = model_name or "llama2"
        return OllamaGenerator(model_name=model_name, **kwargs)

    elif provider == "huggingface":
        model_name = model_name or "google/flan-t5-base"
        return HuggingFaceGenerator(model_name=model_name, **kwargs)

    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: openai, ollama, huggingface"
        )
