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

### Study Guides

- [AI Study Guide](doc/ai-study-guide.md) — Structured learning path from ML foundations to advanced topics
- [Math for AI Notes](doc/math-for-ai-notes.md) — Math study notes (pre-algebra → probability) for AI/ML
- [Agentic AI and the Modern AI Stack](doc/agentic-ai-and-modern-stack.md) — Agentic AI, MCP protocol, and current AI tech landscape

### Tool & Platform Guides

- [Claude Code Guide](doc/claude-code-guide.md) — Features, commands, configuration, and tips for Claude Code
- [Gemma LLM Setup Guide](doc/gemma-llm-setup-guide.md) — Running Gemma locally with Ollama, llama.cpp, and MLX
- [Paperclip Setup Guide](doc/paperclip-setup-guide.md) — AI company orchestration platform setup and usage

### Engineering

- [Eval Harness Engineering](doc/eval-harness-engineering.md) — Building evaluation harnesses to test LLM outputs

### Resources

- [Study Resources](doc/study-resources.md) — Curated links to all learning resources

## Tools

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Local LLM runtime with Metal GPU acceleration |
| [aichat](https://github.com/sigoden/aichat) | Terminal-based LLM chat client |
| [Open WebUI](https://github.com/open-webui/open-webui) | ChatGPT-like web interface for Ollama |
| [Gemma 3](https://ai.google.dev/gemma) | Google's open model family |
