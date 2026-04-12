# Claude Code Guide

A reference for Claude Code features — the CLI/IDE agent by Anthropic.

---

## What Is Claude Code?

Claude Code is an agentic coding assistant that lives in your terminal and IDE.
It can read your codebase, edit files, run commands, search the web, and manage git — all through natural language.

Available as: **CLI** (`claude`), **VS Code extension**, **JetBrains extension**, **Desktop app**, **Web app** (claude.ai/code).

---

## Getting Started

### Install

```bash
npm install -g @anthropic-ai/claude-code
```

### Launch

```bash
# Start in current directory
claude

# Start with a prompt
claude "explain this project"

# Non-interactive (print and exit)
claude -p "what does main.py do"

# Pipe input
cat error.log | claude -p "explain this error"
```

---

## Core Features

### File Operations

Claude Code reads, edits, and creates files directly in your project.

| Action | How It Works |
| ------ | ------------ |
| Read files | Reads any file in your project to understand context |
| Edit files | Makes targeted edits with before/after diffs you approve |
| Create files | Writes new files when needed |
| Search files | Glob patterns and regex search across the codebase |

### Terminal Commands

Claude Code runs shell commands and uses the output.

```
> run the tests
# Claude runs: python3 -m pytest test/ -v
# Shows output, interprets results, fixes failures
```

### Git Integration

| Command | What It Does |
| ------- | ------------ |
| `/commit` | Analyze changes and create a commit with a good message |
| Create PRs | Generate PR title, summary, and push via `gh` |
| Read history | Use `git log`, `git blame` to understand changes |
| Branch management | Create branches, cherry-pick, rebase |

### Web Access

| Tool | Purpose |
| ---- | ------- |
| Web search | Search for docs, error messages, library info |
| Web fetch | Read web pages, documentation, GitHub issues |

---

## Slash Commands

Type these in the Claude Code prompt:

| Command | Description |
| ------- | ----------- |
| `/help` | Show help and available commands |
| `/compact` | Compress conversation to save context |
| `/clear` | Clear conversation history |
| `/commit` | Create a git commit from current changes |
| `/review` | Review code changes |
| `/fast` | Toggle fast mode (same model, faster output) |
| `/model` | Switch model (opus, sonnet, haiku) |
| `/cost` | Show token usage and cost |
| `/memory` | View or edit memory files |
| `/config` | Open settings |
| `/vim` | Toggle vim keybindings |
| `/terminal-setup` | Set up terminal integration (Shift+Enter for newlines) |
| `/mcp` | Show MCP server status |
| `/logout` | Sign out |

---

## Configuration

### CLAUDE.md Files

Project instructions that Claude Code reads automatically. Place them at:

| Location | Scope |
| -------- | ----- |
| `CLAUDE.md` | Project root — applies to all conversations in this repo |
| `src/CLAUDE.md` | Directory-specific — applies when working in `src/` |
| `~/.claude/CLAUDE.md` | Global — applies to all projects |

Use CLAUDE.md for:
- Project structure and conventions
- Build/test commands
- Tech stack details
- Coding style preferences

### Settings Files

| File | Scope | Purpose |
| ---- | ----- | ------- |
| `.claude/settings.json` | Project | MCP servers, permissions, hooks |
| `~/.claude.json` | Global | API keys, default model, global MCP servers |

### Permission Modes

Control what Claude Code can do without asking:

| Mode | Behavior |
| ---- | -------- |
| Default | Asks before edits, commands, and file creation |
| Auto-approve reads | File reads happen automatically |
| Auto-approve edits | File edits happen automatically |
| Yolo mode | Everything runs automatically (use with caution) |

Configure via: `/config` or `~/.claude.json`

---

## MCP Servers

Claude Code supports the Model Context Protocol to connect external tools.

### Configure MCP Servers

In `.claude/settings.json`:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
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

### Check MCP Status

```
/mcp
```

Shows connected servers and available tools.

---

## Hooks

Hooks are shell commands that run automatically in response to Claude Code events.

### Configure in `.claude/settings.json`:

```json
{
  "hooks": {
    "on_tool_use": [
      {
        "tool": "Edit",
        "command": "echo 'File edited: $FILE_PATH'"
      }
    ],
    "on_message": [
      {
        "command": "echo 'Claude responded'"
      }
    ]
  }
}
```

### Hook Events

| Event | Fires When |
| ----- | ---------- |
| `on_tool_use` | Before a tool is executed |
| `on_message` | After Claude sends a message |
| `on_start` | When a conversation begins |

Use hooks for: linting after edits, notifications, logging, custom validation.

---

## Memory System

Claude Code has persistent memory that carries across conversations.

### How It Works

- Memory files are stored in `~/.claude/projects/<project>/memory/`
- `MEMORY.md` is an index loaded into every conversation
- Memory types: user preferences, feedback, project context, references

### Commands

```
> remember that we use pnpm, not npm
# Claude saves to memory

> /memory
# View and edit memory files
```

---

## Agents and Subagents

Claude Code can spawn specialized subagents for parallel work:

| Agent Type | Purpose |
| ---------- | ------- |
| Explore | Fast codebase search and exploration |
| Plan | Design implementation strategies |
| General-purpose | Complex multi-step research tasks |

Subagents run independently and report back results.

---

## IDE Integration

### VS Code

- Install "Claude Code" extension from marketplace
- Opens as a panel in the sidebar
- Shares context with your open files and terminal

### JetBrains

- Install from JetBrains Marketplace
- Works in IntelliJ, PyCharm, WebStorm, etc.

### Features in IDE

- Inline diff view for edits
- Click-to-navigate file references
- Terminal output integration
- Diagnostics awareness (sees linter warnings)

---

## CLI Flags

```bash
# Start interactive session
claude

# One-shot prompt (non-interactive)
claude -p "explain main.py"

# Pipe input
cat file.py | claude -p "review this"

# Resume last conversation
claude --resume

# Use specific model
claude --model sonnet

# Print output as JSON
claude -p "list files" --json

# Run in a specific directory
claude --dir /path/to/project

# Show version
claude --version
```

---

## Tips and Patterns

### Effective Prompting

```
# Be specific
> fix the bug in auth.py where token expiry isn't checked

# Give context
> the tests in test_api.py are failing because the mock is outdated, update them

# Chain tasks
> read the PR comments, fix the issues, then run the tests
```

### Running Commands

```
# Use ! prefix to run a command yourself in the session
! gcloud auth login

# Ask Claude to run commands
> run pytest and fix any failures
```

### Working with Large Codebases

- Use `CLAUDE.md` to describe project structure so Claude navigates faster
- Ask Claude to search before editing: "find where user auth is handled"
- Use `/compact` when the conversation gets long

### Git Workflow

```
> create a branch called fix/auth-expiry
> make the changes to fix token expiry
> /commit
> create a PR
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| -------- | ------ |
| Enter | Send message |
| Shift+Enter | New line (requires terminal setup) |
| Ctrl+C | Cancel current operation |
| Ctrl+L | Clear screen |
| Escape | Cancel current input / interrupt |
| Up/Down | Navigate history |

Run `/terminal-setup` to enable Shift+Enter for multi-line input.

---

## Cost and Usage

```
/cost
```

Shows tokens used and estimated cost for the current session.

| Model | Relative Cost |
| ----- | ------------- |
| Haiku | Lowest |
| Sonnet | Medium |
| Opus | Highest |

Use `/fast` for faster output. Use `/model haiku` for quick, cheap tasks.
