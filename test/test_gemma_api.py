"""Pytest tests for Gemma 3 27B via Ollama API."""

import pytest
import requests

BASE_URL = "http://localhost:11434"
MODEL = "gemma3:27b"


@pytest.fixture(scope="session")
def ollama_available():
    try:
        r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        pytest.skip("Ollama server not running")


def test_server_health(ollama_available):
    """Ollama server responds."""
    r = requests.get(BASE_URL, timeout=5)
    assert r.status_code == 200


def test_model_loaded(ollama_available):
    """Gemma 3 27B model is available."""
    r = requests.get(f"{BASE_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json()["models"]]
    assert any(MODEL in m for m in models)


def test_generate(ollama_available):
    """Generate endpoint returns a response."""
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={"model": MODEL, "prompt": "Say hello.", "stream": False},
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True
    assert len(data["response"]) > 0


def test_chat_completion(ollama_available):
    """Chat completions endpoint works."""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["message"]["content"]) > 0


def test_generate_with_options(ollama_available):
    """Generate with custom temperature and max tokens."""
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": "Count from 1 to 5.",
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 50},
        },
        timeout=60,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["done"] is True


def test_streaming_generate(ollama_available):
    """Streaming generate returns chunked responses."""
    r = requests.post(
        f"{BASE_URL}/api/generate",
        json={"model": MODEL, "prompt": "Say hi.", "stream": True},
        timeout=60,
        stream=True,
    )
    assert r.status_code == 200
    chunks = list(r.iter_lines())
    assert len(chunks) > 0
