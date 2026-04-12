# Agentic AI and the Modern AI Stack (2025-2026)

A guide to the current landscape: AI agents, MCP, and the tools shaping how we build with LLMs.

---

## 1. Agentic AI

### What Makes AI "Agentic"?

Traditional LLMs are stateless — you ask, they answer. Agentic AI goes further:
the model plans, uses tools, observes results, and iterates until the task is done.

```
User Goal
  → Agent plans steps
    → Calls tools (search, code, APIs)
      → Observes results
        → Decides next action
          → Repeats until done
```

### Core Capabilities

| Capability | Description | Example |
| ---------- | ----------- | ------- |
| Planning | Break a goal into subtasks | "Deploy this app" → check deps, write config, test, deploy |
| Tool Use | Call external functions and APIs | Search the web, run code, read files, query databases |
| Memory | Remember context across steps | Track what files were edited, what errors occurred |
| Reflection | Evaluate own output and retry | "That test failed — let me read the error and fix it" |
| Multi-step Reasoning | Chain decisions over many turns | Research → plan → implement → test → iterate |

### Agent Architectures

**ReAct (Reason + Act)**

The most common pattern. The agent alternates between thinking and acting.

```
Thought: I need to find the bug in the auth module
Action: search files for "authenticate" function
Observation: Found in src/auth.py line 42
Thought: Let me read that function
Action: read src/auth.py lines 40-60
Observation: The token check is missing expiry validation
Thought: I'll add the expiry check
Action: edit src/auth.py ...
```

**Plan-and-Execute**

Separates planning from execution. A planner creates the full plan, then an executor runs each step.

```
Planner → [Step 1, Step 2, Step 3, ...]
Executor → runs Step 1 → result → runs Step 2 → result → ...
Planner → revises remaining steps if needed
```

**Multi-Agent**

Multiple specialized agents collaborate on a task.

```
Orchestrator Agent
  ├── Research Agent (searches, reads docs)
  ├── Code Agent (writes and edits code)
  ├── Review Agent (checks quality)
  └── Test Agent (runs tests)
```

### Agentic AI Products

| Product | Type | What It Does |
| ------- | ---- | ------------ |
| Claude Code | Coding agent (CLI/IDE) | Reads codebases, edits files, runs tests, creates PRs |
| Cursor / Windsurf | IDE with agent | AI-powered code editor with inline agent capabilities |
| Devin | Coding agent | Autonomous software engineering agent |
| OpenAI Codex | Coding agent | Cloud-based agent that works in a sandboxed environment |
| ChatGPT Operator | Web agent | Navigates websites and performs tasks on your behalf |
| Google Jules | Coding agent | Async coding agent integrated with GitHub |
| Perplexity | Research agent | Search + synthesis — answers questions with live sources |
| Manus | General agent | Autonomous agent that plans and executes complex tasks |

---

## 2. Model Context Protocol (MCP)

### What Is MCP?

MCP is an open standard (created by Anthropic) that defines how AI applications connect to
external tools and data sources. Think of it as **USB-C for AI** — one protocol,
many connections.

### The Problem MCP Solves

Before MCP, every AI app built custom integrations:

```
Without MCP:                        With MCP:
Claude ←→ custom Slack code         Claude ←→ MCP ←→ Slack Server
Claude ←→ custom GitHub code        Claude ←→ MCP ←→ GitHub Server
Claude ←→ custom DB code            Claude ←→ MCP ←→ Database Server
Cursor ←→ different Slack code      Cursor ←→ MCP ←→ Slack Server (same!)
```

MCP lets any AI client talk to any MCP server with zero custom code.

### Architecture

```
┌─────────────┐     MCP Protocol      ┌─────────────┐
│  MCP Client  │ ◄──────────────────► │  MCP Server  │
│ (Claude, etc)│   JSON-RPC over       │ (tools/data) │
└─────────────┘   stdio or HTTP+SSE   └─────────────┘
```

**Three primitives an MCP server can expose**:

| Primitive | Description | Example |
| --------- | ----------- | ------- |
| Tools | Functions the model can call | `search_files`, `run_query`, `send_message` |
| Resources | Data the model can read | Database schemas, API docs, file contents |
| Prompts | Reusable prompt templates | "Summarize this PR", "Review this code" |

### How MCP Works in Practice

**1. Server exposes capabilities**:

```json
{
  "tools": [
    {
      "name": "search_issues",
      "description": "Search GitHub issues",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": { "type": "string" },
          "repo": { "type": "string" }
        }
      }
    }
  ]
}
```

**2. Client (LLM) discovers and calls tools**:

```
Model sees available tools → decides to call search_issues →
MCP client sends request → MCP server executes → returns results →
Model uses results in response
```

### MCP Server Examples

| Server | What It Provides |
| ------ | ---------------- |
| GitHub | Search repos, read issues/PRs, create branches |
| Slack | Read/send messages, search channels |
| PostgreSQL | Query databases, inspect schemas |
| Filesystem | Read/write local files |
| Puppeteer | Browser automation, screenshots |
| Google Drive | Search and read documents |

### Building an MCP Server (Python)

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def search_notes(query: str) -> str:
    """Search personal notes for a query."""
    # Your search logic here
    results = do_search(query)
    return "\n".join(results)

@mcp.resource("notes://recent")
def recent_notes() -> str:
    """Get the 10 most recent notes."""
    return load_recent_notes()

if __name__ == "__main__":
    mcp.run()
```

### Building an MCP Server (TypeScript)

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({ name: "my-tools", version: "1.0.0" });

server.tool("search_notes", { query: z.string() }, async ({ query }) => {
  const results = await doSearch(query);
  return { content: [{ type: "text", text: results.join("\n") }] };
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

### Using MCP Servers with Claude Code

Add to `.claude/settings.json` or `~/.claude.json`:

```json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["my_mcp_server.py"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
```

---

## 3. Current AI Tech Stack (2025-2026)

### Model Landscape

| Provider | Models | Strengths |
| -------- | ------ | --------- |
| Anthropic | Claude 4.5/4.6 (Opus, Sonnet, Haiku) | Coding, reasoning, safety, long context |
| OpenAI | GPT-4.1, o3, o4-mini | Broad capabilities, reasoning models |
| Google | Gemini 2.5 Pro/Flash | Multimodal, long context (1M tokens) |
| Meta | LLaMA 4 (Scout, Maverick) | Open-weight, local deployment |
| Google | Gemma 3 (1B-27B) | Open-weight, runs locally on your Mac Studio |
| Mistral | Mistral Large, Codestral | Efficient, strong at code |
| DeepSeek | DeepSeek-V3, R1 | Open-weight, strong reasoning |

### Key Techniques

| Technique | What It Is |
| --------- | ---------- |
| Reasoning Models | Models that "think" step-by-step before answering (o3, Claude with extended thinking) |
| Mixture of Experts (MoE) | Only activate a subset of parameters per token — faster inference |
| Long Context | 100K-1M+ token windows — process entire codebases at once |
| Multimodal | Process text, images, audio, video in a single model |
| Structured Output | Models output valid JSON matching a schema (function calling) |
| Streaming | Token-by-token output for responsive UIs |
| Caching | Reuse computed context across requests to save cost and latency |

### Building with LLMs

| Layer | Tools |
| ----- | ----- |
| Model APIs | Anthropic API, OpenAI API, Google AI Studio, Ollama (local) |
| SDKs | `anthropic` (Python/TS), `openai` (Python/TS), Vercel AI SDK |
| Orchestration | LangChain, LlamaIndex, Haystack, Semantic Kernel |
| Agent Frameworks | Claude Agent SDK, OpenAI Agents SDK, CrewAI, AutoGen, LangGraph |
| RAG / Search | ChromaDB, Pinecone, Weaviate, Qdrant, pgvector |
| Evaluation | Braintrust, Promptfoo, LangSmith, custom eval harnesses |
| Deployment | Modal, Replicate, Together AI, Fireworks, vLLM |
| Local Inference | Ollama, llama.cpp, MLX, LM Studio |

### AI-Powered Developer Tools

| Tool | Category | Description |
| ---- | -------- | ----------- |
| Claude Code | CLI/IDE agent | Agentic coding assistant with terminal + IDE integration |
| GitHub Copilot | Code completion | Inline suggestions and chat inside VS Code |
| Cursor | AI IDE | Fork of VS Code with deep agent integration |
| Windsurf | AI IDE | Agent-powered IDE by Codeium |
| v0 | UI generation | Generate React/Next.js UI from prompts (Vercel) |
| Bolt / Lovable | App generation | Generate full-stack apps from natural language |
| Cline | IDE agent | Open-source coding agent for VS Code |

### Prompt Engineering Evolved

| Pattern | Description |
| ------- | ----------- |
| System Prompts | Define model behavior, constraints, and persona |
| Few-Shot Examples | Show input/output pairs to guide format |
| Chain of Thought | "Think step by step" — improves reasoning |
| Tool Use | Model decides when and how to call external functions |
| Retrieval-Augmented | Inject relevant context from a knowledge base |
| Multi-Turn | Maintain conversation state for complex tasks |
| Structured Output | Request JSON/XML output matching a schema |

---

## 4. Hands-On Projects

Build these to get practical experience with the current stack:

### Beginner

1. **MCP Server for local notes** — Build a simple MCP server that searches your local markdown files
2. **Chat with tools** — Give an LLM access to a calculator and web search via function calling
3. **Structured data extraction** — Use an LLM to extract JSON from unstructured text

### Intermediate

4. **RAG chatbot** — Index your docs folder, answer questions with retrieved context
5. **Code review agent** — Agent that reads a diff, checks for issues, suggests fixes
6. **Multi-tool agent** — Build a ReAct agent with file read/write, search, and shell access

### Advanced

7. **Custom MCP server ecosystem** — Build MCP servers for your own tools and connect them to Claude Code
8. **Multi-agent pipeline** — Orchestrate research + coding + review agents on a task
9. **Eval harness** — Systematically compare models/prompts on your own benchmark

---

## 5. Resources

### Documentation

- [MCP Specification](https://modelcontextprotocol.io) — Official MCP docs and spec
- [Anthropic Docs](https://docs.anthropic.com) — Claude API, tool use, prompt engineering
- [Claude Code Docs](https://docs.anthropic.com/en/docs/claude-code) — Agent setup, MCP, hooks

### Learning

- [Anthropic Courses](https://github.com/anthropics/courses) — Free prompt engineering and tool use courses
- [DeepLearning.AI](https://www.deeplearning.ai) — Short courses on agents, RAG, fine-tuning
- [Simon Willison's Blog](https://simonwillison.net) — Practical LLM insights and tool coverage
- [Latent Space Podcast](https://www.latent.space) — AI engineering interviews and analysis

### Repositories

- [MCP Servers](https://github.com/modelcontextprotocol/servers) — Official and community MCP servers
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) — Build custom agents with Claude
- [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) — Curated MCP server list
