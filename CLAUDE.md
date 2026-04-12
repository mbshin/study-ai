# study-ai

Local LLM study and experimentation project on Mac Studio (M1 Max, 32 GB).

## Project Structure

```
doc/     — documentation and study guides
docker/  — Docker Compose files (Open WebUI)
test/    — pytest test files
```

## Study Guides

- `doc/ai-study-guide.md` — Structured AI/ML learning path (foundations → advanced)
- `doc/agentic-ai-and-modern-stack.md` — Agentic AI, MCP, and current AI tech stack
- `doc/claude-code-guide.md` — Claude Code features, commands, and configuration
- `doc/eval-harness-engineering.md` — Building eval harnesses to test LLM outputs
- `doc/gemma-llm-setup-guide.md` — Local Gemma LLM setup on Mac Studio
- `doc/study-resources.md` — Curated links to all learning resources

## Local LLM Setup

- **Ollama** running as a background service (`brew services start ollama`)
- **Gemma 3 27B** is the primary model (`ollama run gemma3:27b`)
- Ollama API: `http://localhost:11434`
- OpenAI-compatible endpoint: `http://localhost:11434/v1`

## CLI Tools

- **aichat** — terminal LLM client connected to Ollama
  - Config: `~/Library/Application Support/aichat/config.yaml`
  - Default model: `ollama:gemma3:27b`
  - Usage: `aichat "question"`, `aichat -s` for interactive session, `aichat -f file.py "review"`

## Web UI

- **Open WebUI** — ChatGPT-like browser interface at `http://localhost:3000`
  - Run: `cd docker && docker compose up -d`
  - Stop: `cd docker && docker compose down`

## Testing

```bash
python3 -m pytest test/ -v
```

Tests use `requests` against the local Ollama API. Ollama must be running.

## Dependencies

- Python 3.14+
- pytest, requests (`pip3 install pytest requests`)
- ollama (`brew install ollama`)
- aichat (`brew install aichat`)
- Docker (for Open WebUI)
