# Eval Harness Engineering

How to systematically evaluate LLM outputs — build confidence before shipping prompts, models, or agents.

---

## Why Eval Harnesses?

"Vibes-based" testing (manually checking a few outputs) doesn't scale. An eval harness lets you:

- **Catch regressions** — did changing the prompt break something?
- **Compare models** — is Gemma 27B better than 12B for your task?
- **Measure improvements** — did RAG actually help?
- **Build confidence** — ship changes knowing they work across many cases

---

## Core Concepts

| Concept | Description |
| ------- | ----------- |
| Eval Dataset | A set of test inputs, optionally with expected outputs |
| Metric | How you score each output (accuracy, similarity, pass/fail) |
| Judge | What does the scoring — code, human, or another LLM |
| Harness | The system that runs inputs through a model and collects scores |
| Baseline | A reference run to compare against (before your change) |

---

## Anatomy of an Eval Harness

```
┌──────────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│ Eval Dataset  │ ──► │  Run Model │ ──► │  Judge    │ ──► │  Report   │
│ (inputs +     │     │  (get      │     │  (score   │     │  (compare │
│  expected)    │     │   outputs) │     │   each)   │     │   runs)   │
└──────────────┘     └───────────┘     └──────────┘     └──────────┘
```

---

## Step 1: Build an Eval Dataset

### Format

Keep it simple — JSON Lines works well:

```jsonl
{"input": "What is the capital of France?", "expected": "Paris"}
{"input": "Translate 'hello' to Japanese", "expected": "こんにちは"}
{"input": "Is 17 a prime number?", "expected": "Yes"}
```

For more complex tasks, use structured fields:

```jsonl
{"input": "Summarize this article", "context": "...", "expected_keywords": ["climate", "policy"], "max_length": 200}
```

### How Many Examples?

| Stage | Count | Purpose |
| ----- | ----- | ------- |
| Prototyping | 10-20 | Quick sanity checks |
| Development | 50-100 | Catch common failures |
| Pre-ship | 200+ | Statistical confidence |

### Where to Get Test Cases

- Write them manually from real use cases
- Sample from production logs (anonymized)
- Ask the LLM to generate edge cases, then curate
- Use existing benchmarks as a starting point

---

## Step 2: Choose Your Metrics

### Exact Match

Simplest — is the output exactly right?

```python
def exact_match(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()
```

Good for: factual Q&A, classification, structured output.

### Contains / Keyword Match

Does the output contain key information?

```python
def keyword_match(output: str, keywords: list[str]) -> float:
    found = sum(1 for kw in keywords if kw.lower() in output.lower())
    return found / len(keywords)
```

Good for: summaries, open-ended answers.

### LLM-as-Judge

Use another LLM to score the output. Most flexible approach.

```python
def llm_judge(input: str, output: str, criteria: str) -> int:
    prompt = f"""Rate this response from 1-5.

Question: {input}
Response: {output}
Criteria: {criteria}

Return only the number."""

    response = call_llm(prompt)
    return int(response.strip())
```

Good for: quality, helpfulness, safety — anything subjective.

### Code-Based Checks

For structured outputs, validate programmatically:

```python
import json

def valid_json(output: str) -> bool:
    try:
        data = json.loads(output)
        return "name" in data and "age" in data  # schema check
    except json.JSONDecodeError:
        return False
```

### Metric Summary

| Metric | Best For | Effort |
| ------ | -------- | ------ |
| Exact match | Factual, classification | Low |
| Keyword match | Summaries, open-ended | Low |
| Regex match | Structured formats | Low |
| Code validation | JSON, code output | Medium |
| Similarity (embedding) | Semantic equivalence | Medium |
| LLM-as-judge | Quality, safety, style | High |
| Human review | Final validation | Highest |

---

## Step 3: Build the Harness

### Minimal Python Harness

```python
import json
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:27b"


def call_model(prompt: str) -> str:
    """Call local Ollama model."""
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    })
    return resp.json()["response"]


def load_dataset(path: str) -> list[dict]:
    """Load JSONL eval dataset."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_eval(dataset_path: str, judge_fn) -> dict:
    """Run eval and return results."""
    dataset = load_dataset(dataset_path)
    results = []

    for i, example in enumerate(dataset):
        output = call_model(example["input"])
        score = judge_fn(output, example)
        results.append({
            "input": example["input"],
            "output": output,
            "expected": example.get("expected"),
            "score": score,
        })
        print(f"[{i+1}/{len(dataset)}] score={score}")

    scores = [r["score"] for r in results]
    summary = {
        "total": len(results),
        "mean_score": sum(scores) / len(scores),
        "pass_rate": sum(1 for s in scores if s >= 1.0) / len(scores),
        "results": results,
    }
    return summary


# --- Example judge function ---
def exact_judge(output: str, example: dict) -> float:
    expected = example.get("expected", "")
    return 1.0 if expected.lower() in output.lower() else 0.0


if __name__ == "__main__":
    summary = run_eval("eval_dataset.jsonl", exact_judge)
    print(f"\nResults: {summary['total']} examples, "
          f"mean={summary['mean_score']:.2f}, "
          f"pass={summary['pass_rate']:.0%}")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
```

### Running It

```bash
# Create a small dataset
cat > eval_dataset.jsonl << 'EOF'
{"input": "What is the capital of France?", "expected": "paris"}
{"input": "What is 2 + 2?", "expected": "4"}
{"input": "What language is spoken in Brazil?", "expected": "portuguese"}
EOF

# Run eval
python3 eval_harness.py
```

---

## Step 4: Compare Runs

The real value comes from comparing before/after:

```python
def compare_runs(baseline: dict, current: dict) -> None:
    """Compare two eval runs."""
    b_score = baseline["mean_score"]
    c_score = current["mean_score"]
    diff = c_score - b_score

    print(f"Baseline: {b_score:.3f}")
    print(f"Current:  {c_score:.3f}")
    print(f"Delta:    {diff:+.3f} ({'improved' if diff > 0 else 'regressed'})")

    # Find regressions
    for b, c in zip(baseline["results"], current["results"]):
        if c["score"] < b["score"]:
            print(f"\nRegression: {b['input'][:60]}...")
            print(f"  Was: {b['score']} → Now: {c['score']}")
```

---

## Step 5: Iterate

### What to Vary

| Variable | Example Changes |
| -------- | --------------- |
| Prompt | Add system prompt, few-shot examples, chain-of-thought |
| Model | gemma3:27b vs gemma3:12b, different quantizations |
| Temperature | 0.0 (deterministic) vs 0.7 (creative) |
| Context | With vs without RAG, different chunk sizes |
| Post-processing | Parse JSON, extract answer, clean whitespace |

### Eval-Driven Development Loop

```
1. Write eval dataset for your task
2. Run baseline (current prompt + model)
3. Make one change (prompt, model, or pipeline)
4. Run eval again
5. Compare — did it improve?
6. Keep the change if yes, revert if no
7. Repeat
```

---

## Existing Eval Tools

| Tool | Description | Local-Friendly |
| ---- | ----------- | -------------- |
| pytest | Write evals as test cases — simple and familiar | Yes |
| Promptfoo | YAML-based eval framework, supports Ollama | Yes |
| lm-eval-harness | EleutherAI's benchmark suite (MMLU, etc.) | Yes |
| Braintrust | Eval platform with logging and comparison | Cloud |
| LangSmith | LangChain's tracing and eval platform | Cloud |
| DeepEval | Python framework with built-in metrics | Yes |

### Using pytest as an Eval Harness

```python
# test/test_eval.py
import requests
import pytest

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": "gemma3:27b",
        "prompt": prompt,
        "stream": False,
    })
    return resp.json()["response"]

@pytest.mark.parametrize("question,expected", [
    ("What is the capital of France?", "Paris"),
    ("What is 7 * 8?", "56"),
    ("Is the Earth flat?", "No"),
])
def test_factual_qa(question, expected):
    answer = ask(question)
    assert expected.lower() in answer.lower(), f"Expected '{expected}' in: {answer[:100]}"
```

```bash
python3 -m pytest test/test_eval.py -v
```

---

## Tips

- **Start small** — 10 hand-picked examples beat 1000 auto-generated ones
- **Include edge cases** — empty input, very long input, ambiguous questions, adversarial prompts
- **Version your datasets** — commit them to git, track changes over time
- **Use temperature 0** — for reproducible evals, set `temperature: 0`
- **Separate correctness from style** — score facts and formatting independently
- **Automate the loop** — run evals in CI or as a pre-commit check
- **Log everything** — save full inputs, outputs, and scores for debugging
