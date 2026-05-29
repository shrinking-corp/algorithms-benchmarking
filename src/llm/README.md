# llm/

This module provides abstractions and implementations for interacting with various Large Language Model (LLM) APIs.

## Contents
- `base.py`: Abstract base classes for LLM clients (Template Method pattern).
- `openai_client.py`: OpenAI-compatible API client.
- `ollama_client.py`: Local Ollama API client.
- `litellm_client.py`: LiteLLM proxy client.
- `proxy.py`: CachingProxy for disk/in-memory caching of LLM responses (Proxy pattern).
- `factory.py`: Factory for creating LLM clients based on configuration.

## Features
- Unified interface for multiple LLM backends
- Transparent disk and in-memory caching for fast, cost-effective benchmarking
- Easy extension to new LLM providers

## Design Patterns
- **Template Method:** For API call workflow in base client.
- **Proxy:** For caching LLM responses.
- **Factory:** For client instantiation.

LLM clients are used by the benchmarking orchestrator to generate code from PlantUML diagrams. Caching ensures efficient repeated runs.