# study-ai

Local LLM study and experimentation project on Mac Studio (Apple M1 Max, 32 GB).

## Setup

### Prerequisites

- macOS with Apple Silicon
- [Homebrew](https://brew.sh)

### Install

```bash
# Ollama — local LLM runtime
brew install ollama
brew services start ollama
ollama pull gemma3:27b

# aichat — CLI chat client
brew install aichat

# Python test dependencies
pip3 install pytest requests

# Open WebUI (requires Docker)
cd docker && docker compose up -d
```

### Run Tests

```bash
python3 -m pytest test/ -v
```

## Project Structure

```text
doc/     — setup guides and documentation
docker/  — Docker Compose files (Open WebUI)
test/    — API tests (pytest)
```

## Documentation

- [Gemma LLM Setup Guide](doc/gemma-llm-setup-guide.md) — hardware recommendations, Ollama/llama.cpp/Python/aichat setup

## Tools

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Local LLM runtime with Metal GPU acceleration |
| [aichat](https://github.com/sigoden/aichat) | Terminal-based LLM chat client |
| [Open WebUI](https://github.com/open-webui/open-webui) | ChatGPT-like web interface for Ollama |
| [Gemma 3](https://ai.google.dev/gemma) | Google's open model family |
