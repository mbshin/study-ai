# AI Study Guide

A structured learning path from fundamentals to hands-on practice with local LLMs.

---

## Phase 1: Foundations

### 1.1 Machine Learning Basics

| Topic | Key Concepts |
|-------|-------------|
| Supervised Learning | Regression, classification, loss functions, gradient descent |
| Unsupervised Learning | Clustering (k-means), dimensionality reduction (PCA) |
| Evaluation | Train/test split, cross-validation, overfitting, bias-variance tradeoff |
| Optimization | SGD, Adam, learning rate scheduling |

**Recommended Resources**:
- Andrew Ng — Machine Learning Specialization (Coursera)
- fast.ai — Practical Deep Learning for Coders
- 3Blue1Brown — Neural Networks (YouTube)

### 1.2 Deep Learning Fundamentals

| Topic | Key Concepts |
|-------|-------------|
| Neural Networks | Perceptrons, activation functions (ReLU, sigmoid), backpropagation |
| CNNs | Convolution, pooling, image classification |
| RNNs / LSTMs | Sequence modeling, vanishing gradients |
| Regularization | Dropout, batch normalization, weight decay |

### 1.3 Math for AI (Intuition-First Approach)

You don't need to be a math expert to learn AI. Focus on building intuition, not proofs.
Use visual and interactive resources — skip the textbook-heavy approach.

**What to learn (and what to skip for now)**:

| Topic | What You Need | Skip For Now |
|-------|---------------|--------------|
| Linear Algebra | Vectors as lists of numbers, matrix multiply as transformation, dot product as similarity | Eigenvalues, proofs, abstract spaces |
| Calculus | Gradient = "which direction to improve", chain rule = "passing blame backwards" | Formal limits, integration, proofs |
| Probability | Bayes = "updating beliefs with evidence", distributions = "shapes of randomness" | Measure theory, moment generating functions |
| Statistics | Mean, variance, correlation — enough to read training metrics | Hypothesis testing, p-values |

**Beginner-Friendly Math Resources**:

- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Visual, no prerequisites
- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — Intuition over formulas
- [StatQuest (YouTube)](https://www.youtube.com/@statquest) — Statistics and ML explained simply
- [Seeing Theory](https://seeing-theory.brown.edu/) — Interactive probability visualizations
- Khan Academy — Fill in specific gaps as needed (free)

**Practical tip**: Learn math *as you need it*. When you encounter a concept in an AI tutorial
(e.g., "cross-entropy loss"), look up just that concept. Don't try to complete a math course first.

---

## Phase 2: Natural Language Processing

### 2.1 Text Processing Foundations

- Tokenization (word, subword, BPE, SentencePiece)
- Embeddings (Word2Vec, GloVe, contextual embeddings)
- Language model basics (n-gram, neural LMs)

### 2.2 The Transformer Architecture

The core building block of modern LLMs.

| Component | Purpose |
|-----------|---------|
| Self-Attention | Lets each token attend to all other tokens in the sequence |
| Multi-Head Attention | Parallel attention heads capture different relationships |
| Positional Encoding | Injects token order information (sinusoidal or RoPE) |
| Feed-Forward Network | Processes each position independently after attention |
| Layer Normalization | Stabilizes training across layers |
| Residual Connections | Enables gradient flow in deep networks |

**Key Papers**:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

### 2.3 Large Language Models

| Concept | Description |
|---------|-------------|
| Pre-training | Next-token prediction on massive text corpora |
| Fine-tuning | Adapting a base model to specific tasks or domains |
| Instruction Tuning | Training the model to follow human instructions |
| RLHF | Reinforcement Learning from Human Feedback for alignment |
| DPO | Direct Preference Optimization — simpler alternative to RLHF |
| Scaling Laws | Model performance improves predictably with compute, data, and parameters |

**Model Families to Study**:
- **GPT** (OpenAI) — decoder-only, autoregressive
- **Gemma** (Google) — open-weight, efficient architecture
- **LLaMA** (Meta) — open-weight, widely used in research
- **Claude** (Anthropic) — focus on safety and helpfulness
- **Mistral / Mixtral** — efficient, mixture-of-experts

---

## Phase 3: Practical LLM Skills

### 3.1 Running Models Locally

This project uses Ollama + Gemma 3 27B on Mac Studio (M1 Max, 32 GB).
See [gemma-llm-setup-guide.md](gemma-llm-setup-guide.md) for full setup.

```bash
# Start Ollama
brew services start ollama

# Pull and run model
ollama run gemma3:27b

# API call
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:27b",
  "prompt": "What is attention in transformers?",
  "stream": false
}'
```

### 3.2 Prompt Engineering

| Technique | Description | Example |
|-----------|-------------|---------|
| Zero-shot | Direct question, no examples | "Classify this sentiment: ..." |
| Few-shot | Provide examples before the question | "Positive: great! / Negative: terrible! / Classify: ..." |
| Chain-of-Thought | Ask the model to reason step by step | "Think step by step..." |
| System Prompts | Set model behavior and constraints | "You are a Python expert..." |
| ReAct | Reason + Act — combine reasoning with tool use | Thought → Action → Observation loop |

### 3.3 Quantization

How large models fit on consumer hardware.

| Format | Bits | Quality | Memory (27B) |
|--------|------|---------|-------------|
| FP16 | 16 | Full | ~54 GB |
| Q8_0 | 8 | Near-full | ~28 GB |
| Q4_K_M | 4 | Good | ~18 GB |
| Q2_K | 2 | Degraded | ~10 GB |

Key formats: **GGUF** (llama.cpp/Ollama), **GPTQ**, **AWQ**, **MLX** (Apple Silicon).

### 3.4 API Integration

```python
# Using Ollama's OpenAI-compatible API
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")

response = client.chat.completions.create(
    model="gemma3:27b",
    messages=[{"role": "user", "content": "Explain attention mechanism"}],
)
print(response.choices[0].message.content)
```

---

## Phase 4: Advanced Topics

### 4.1 Retrieval-Augmented Generation (RAG)

Augment LLM responses with external knowledge.

```
Query → Embed → Search vector DB → Retrieve relevant docs → LLM generates answer with context
```

| Component | Options |
|-----------|---------|
| Embedding Model | `nomic-embed-text`, `mxbai-embed-large` (via Ollama) |
| Vector Database | ChromaDB, FAISS, Qdrant, Weaviate |
| Chunking | Fixed-size, recursive, semantic splitting |
| Reranking | Cross-encoder models for relevance scoring |

### 4.2 Fine-Tuning

| Method | Description | Use Case |
|--------|-------------|----------|
| Full Fine-Tune | Update all parameters | Maximum quality, needs large GPU |
| LoRA | Low-rank adapter matrices | Efficient, fits on consumer hardware |
| QLoRA | LoRA on quantized model | Fine-tune large models with limited memory |
| Prefix Tuning | Learn soft prompt tokens | Lightweight task adaptation |

Tools: **Unsloth** (fast LoRA), **Hugging Face TRL**, **Axolotl**, **MLX fine-tuning**

### 4.3 Agents and Tool Use

LLMs that can take actions, not just generate text.

| Concept | Description |
|---------|-------------|
| Function Calling | LLM outputs structured tool invocations |
| Planning | Breaking complex tasks into subtasks |
| Memory | Short-term (context) and long-term (external storage) |
| Multi-Agent | Multiple specialized LLMs collaborating |

Frameworks: **LangChain**, **LlamaIndex**, **CrewAI**, **Claude Agent SDK**

### 4.4 Evaluation and Benchmarks

| Benchmark | Measures |
|-----------|---------|
| MMLU | Broad knowledge across 57 subjects |
| HumanEval | Code generation accuracy |
| HellaSwag | Commonsense reasoning |
| GSM8K | Grade-school math |
| MT-Bench | Multi-turn conversation quality |
| Arena ELO | Human preference ranking (Chatbot Arena) |

### 4.5 Safety and Alignment

- Constitutional AI (Anthropic)
- Red teaming and adversarial testing
- Hallucination detection and mitigation
- Bias evaluation and fairness

---

## Phase 5: Hands-On Projects

Progress through these to build practical skills:

1. **Chat with local LLM** — Use Ollama API to build a simple chatbot (Python/CLI)
2. **Document Q&A (RAG)** — Index local files, answer questions with retrieved context
3. **Code Review Assistant** — Feed code to LLM and get structured feedback
4. **Summarization Pipeline** — Process and summarize long documents with chunking
5. **Fine-tune a small model** — LoRA fine-tune Gemma 3 4B on a custom dataset
6. **Multi-tool Agent** — Build an agent that uses search, calculator, and code execution
7. **Evaluation Harness** — Compare model outputs systematically with scoring

---

## Study Log

Track progress here:

| Date | Topic | Notes |
|------|-------|-------|
| | | |

---

## Quick Reference Links

- [Ollama](https://ollama.com) — Local model runner
- [Hugging Face](https://huggingface.co) — Models, datasets, spaces
- [Gemma](https://ai.google.dev/gemma) — Google's open models
- [Papers With Code](https://paperswithcode.com) — Research papers + benchmarks
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual explanation
- [Andrej Karpathy — YouTube](https://www.youtube.com/@AndrejKarpathy) — Neural nets from scratch
- [LLM Visualization](https://bbycroft.net/llm) — Interactive 3D transformer walkthrough
