# study-ai

Local LLM study and experimentation project on Mac Studio (M1 Max, 32 GB).

## Project Structure

```
doc/     — documentation and setup guides
test/    — pytest test files
```

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
