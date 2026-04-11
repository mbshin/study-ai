# Gemma LLM Setup Guide — Mac Studio

## Hardware Spec

| Item | Detail |
|------|--------|
| Model | Mac Studio (Mac13,1) |
| Chip | Apple M1 Max |
| CPU Cores | 10 (8P + 2E) |
| GPU Cores | 24 |
| Unified Memory | 32 GB |
| Metal Support | Metal 4 |
| OS | macOS (Darwin 25.3.0) |

## Gemma Model Size Recommendations

With 32 GB unified memory, the following Gemma variants fit comfortably:

| Model | Parameters | Memory Required (approx.) | Fit? |
|-------|-----------|--------------------------|------|
| Gemma 3 1B | 1B | ~2 GB | Yes |
| Gemma 3 4B | 4B | ~4 GB | Yes |
| Gemma 3 12B | 12B | ~9 GB | Yes |
| Gemma 3 27B | 27B | ~18 GB | Yes |
| Gemma 3 27B (Q8) | 27B | ~28 GB | Tight but possible |

> **Recommendation**: Gemma 3 27B (Q4_K_M quantization) is the sweet spot for this machine — strong quality while leaving headroom for the system.

## Option 1: Ollama (Recommended)

Ollama is the simplest way to run Gemma locally on macOS with Metal GPU acceleration.

### Install

```bash
brew install ollama
```

### Start Ollama server

```bash
ollama serve
```

### Pull and run Gemma

```bash
# Gemma 3 — choose a size
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull gemma3:12b
ollama pull gemma3:27b      # recommended for this hardware

# Run interactively
ollama run gemma3:27b
```

### API usage

Ollama exposes a local REST API at `http://localhost:11434`:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:27b",
  "prompt": "Explain transformer architecture briefly.",
  "stream": false
}'
```

### OpenAI-compatible endpoint

```bash
curl http://localhost:11434/v1/chat/completions -d '{
  "model": "gemma3:27b",
  "messages": [{"role": "user", "content": "Hello"}]
}'
```

This works with any OpenAI SDK client by pointing `base_url` to `http://localhost:11434/v1`.

## Option 2: llama.cpp (Direct)

For more control over quantization and inference parameters.

### Install

```bash
brew install llama.cpp
```

### Download model

Download GGUF files from Hugging Face (e.g., `bartowski/gemma-3-27b-it-GGUF`):

```bash
# Example: download Q4_K_M quantization
huggingface-cli download bartowski/gemma-3-27b-it-GGUF \
  gemma-3-27b-it-Q4_K_M.gguf \
  --local-dir ./models
```

### Run

```bash
llama-server \
  -m ./models/gemma-3-27b-it-Q4_K_M.gguf \
  -c 8192 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

- `-ngl 99`: offload all layers to Metal GPU
- `-c 8192`: context window size

## Option 3: Python (transformers + MLX)

### Using MLX (Apple-optimized)

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-3-27b-it-4bit")
response = generate(model, tokenizer, prompt="Hello, Gemma!", max_tokens=256)
print(response)
```

### Using Hugging Face Transformers

```bash
pip install transformers torch accelerate
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("What is attention?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> Note: For 27B on transformers, you will need quantization (bitsandbytes) to fit in 32 GB. MLX is preferred on Apple Silicon.

## CLI Tool: aichat

aichat is a terminal-based LLM client that connects to Ollama — the closest experience to Claude Code for local models.

### Install

```bash
brew install aichat
```

### Configure

Config file: `~/Library/Application Support/aichat/config.yaml`

```yaml
model: ollama:gemma3:27b
clients:
  - type: openai-compatible
    name: ollama
    api_base: http://localhost:11434/v1
    api_key: null
    models:
      - name: gemma3:27b
        max_input_tokens: 128000
      - name: gemma3:12b
        max_input_tokens: 128000
      - name: gemma3:4b
        max_input_tokens: 128000
```

### Usage

```bash
# Single question
aichat "explain transformers"

# Pipe a file in
cat file.py | aichat "explain this code"

# Include a file
aichat -f file.py "review this"

# Interactive session
aichat -s

# Generate code only
aichat -c "python fibonacci"

# Execute shell commands via natural language
aichat -e "list all python files"

# Switch model
aichat -m ollama:gemma3:4b "quick question"
```

## Performance Tips for M1 Max

1. **Use Metal GPU**: Both Ollama and llama.cpp automatically use Metal on macOS — no extra config needed.
2. **Quantization**: Q4_K_M provides the best quality-to-memory ratio. Q8 is possible for 12B and below.
3. **Context length**: Longer context uses more memory. Start with 4096-8192 and increase if needed.
4. **Close memory-heavy apps**: Unified memory is shared between CPU and GPU — free up RAM for larger models.
5. **Monitor memory**: Use `Activity Monitor` or `sudo powermetrics --samplers gpu_power` to watch GPU utilization.

## Quick Verification

After setup, verify GPU acceleration is active:

```bash
# Ollama — check logs for Metal
ollama run gemma3:27b "Say hello" 2>&1 | head

# llama.cpp — look for "Metal" in startup output
# Should show: ggml_metal_init: found device: Apple M1 Max
```

Expected throughput on M1 Max (32 GB) with Gemma 3 27B Q4:
- **Prompt processing**: ~50-80 tokens/sec
- **Generation**: ~15-25 tokens/sec
