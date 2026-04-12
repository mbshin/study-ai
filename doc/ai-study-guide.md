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

### 1.3 Math for AI (From the Ground Up)

A step-by-step math path starting from basics. Each level builds on the previous one.
Use visual resources and practice problems — not textbooks.

#### Level 1: Arithmetic and Pre-Algebra (Start Here)

If math feels rusty, start here with no shame. This is the foundation everything else sits on.

| Topic | Why It Matters for AI |
| ----- | --------------------- |
| Fractions, decimals, percentages | Probabilities are expressed this way (0.95 = 95%) |
| Negative numbers, absolute value | Loss values, gradients can be negative |
| Exponents and logarithms | Exponentials appear everywhere (softmax, log-loss) |
| Order of operations | Reading and writing formulas correctly |
| Basic graphing (x-y plots) | Every training chart is a graph |

**Resources**: [Khan Academy — Pre-Algebra](https://www.khanacademy.org/math/pre-algebra) (free, self-paced)

#### Level 2: Algebra

| Topic | Why It Matters for AI |
| ----- | --------------------- |
| Variables and equations | Model parameters are variables being solved for |
| Functions (f(x) = ...) | Neural networks are nested functions |
| Slopes and lines (y = mx + b) | Linear regression is literally fitting a line |
| Systems of equations | Multiple constraints, multiple unknowns |
| Summation notation (Σ) | Loss functions sum over all examples |

**Resources**: [Khan Academy — Algebra 1 & 2](https://www.khanacademy.org/math/algebra) (free)

#### Level 3: Linear Algebra (Essential for AI)

This is the language neural networks speak — all data flows through matrices.

| Topic | Why It Matters for AI |
| ----- | --------------------- |
| Vectors | A single data point = a vector (list of numbers) |
| Matrices | A batch of data = a matrix, model weights = matrices |
| Dot product | Measures similarity — core of attention mechanism |
| Matrix multiplication | Every neural network layer is a matrix multiply |
| Transpose | Reshaping data and weights |

**Resources**:

- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — Visual, no prerequisites
- [Khan Academy — Linear Algebra](https://www.khanacademy.org/math/linear-algebra) — Practice problems
- [Interactive Linear Algebra](https://textbooks.math.gatech.edu/ila/) — Free online textbook with visuals

#### Level 4: Calculus (Gradient = How to Improve)

You don't need all of calculus — just enough to understand how models learn.

| Topic | Why It Matters for AI |
| ----- | --------------------- |
| Derivatives (rate of change) | "How much does the output change when I change this weight?" |
| Chain rule | Backpropagation = chain rule applied through layers |
| Partial derivatives | Models have millions of weights, each with its own gradient |
| Gradient = vector of partials | Points in the direction of steepest improvement |
| Minima and maxima | Training = finding the minimum of the loss function |

**Resources**:

- [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) — Intuition over formulas
- [Khan Academy — Calculus](https://www.khanacademy.org/math/calculus-1) — Step-by-step practice

#### Level 5: Probability and Statistics

| Topic | Why It Matters for AI |
| ----- | --------------------- |
| Probability basics | Every prediction is a probability |
| Conditional probability | "What's the chance of X given Y?" — the core of language models |
| Bayes' theorem | Updating beliefs with new evidence |
| Mean, variance, std deviation | Reading training metrics and evaluating models |
| Distributions (normal, uniform) | Data and weight initialization follow distributions |
| Cross-entropy | The most common loss function for classification |

**Resources**:

- [StatQuest (YouTube)](https://www.youtube.com/@statquest) — Statistics and ML explained simply
- [Seeing Theory](https://seeing-theory.brown.edu/) — Interactive probability visualizations
- [Khan Academy — Statistics & Probability](https://www.khanacademy.org/math/statistics-probability)

#### Suggested Study Order

```text
Level 1 (1-2 weeks) → Level 2 (2-3 weeks) → Level 3 (3-4 weeks) → Level 4 (2-3 weeks) → Level 5 (2-3 weeks)
```

You can start Phase 2 (NLP concepts) while working through Levels 3-5.
Start Phase 3 (hands-on LLM work) anytime — practical experience and math study reinforce each other.

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
