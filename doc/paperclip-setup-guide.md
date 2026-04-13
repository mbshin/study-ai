# Paperclip — AI Company Orchestration Platform

## What is Paperclip?

Paperclip is an open-source platform for orchestrating **zero-human companies** — it coordinates multiple AI agents toward business objectives. Think of it as the management layer that employs, organizes, and supervises autonomous AI agents.

- **Repository**: <https://github.com/paperclipai/paperclip>
- **Docs**: <https://paperclip.ing/docs>
- **Discord**: <https://discord.gg/m4HZY7xNG3>
- **License**: MIT

### What It Does

| Capability | Description |
|---|---|
| Multi-Agent Coordination | Manages agents from Claude Code, Codex, Cursor, Bash, HTTP, and more |
| Org Structure | Org charts with roles, reporting lines, and hierarchies |
| Goal Alignment | Tasks trace back to company missions via goal inheritance |
| Cost Management | Monthly budgets per agent with automatic throttling |
| Heartbeat System | Scheduled agent activation with persistent state across sessions |
| Governance | Human approval gates, rollback, versioned config changes |
| Multi-Company | Single deployment hosts multiple isolated companies |
| Audit Logging | Full conversation tracing and immutable decision records |
| Skill Injection | Agents learn workflows dynamically at runtime |
| Company Portability | Export/import orgs with automatic secret scrubbing |

### What It Is NOT

- Not a chatbot or prompt management system
- Not an agent-building framework
- Not a drag-and-drop workflow tool
- Not for single-agent use cases

## Prerequisites

| Dependency | Version |
|---|---|
| Node.js | 20+ |
| pnpm | 9.15+ |

PostgreSQL is embedded automatically for local use (external PostgreSQL supported for production). Docker is optional.

### Install pnpm (if needed)

```bash
brew install pnpm
```

Or via corepack:

```bash
corepack enable
corepack prepare pnpm@latest --activate
```

## Quick Setup

The fastest way to get started:

```bash
npx paperclipai onboard --yes
```

The server starts at **http://localhost:3100**.

### Network Binding Options

```bash
# LAN access (other devices on your network)
npx paperclipai onboard --yes --bind lan

# Tailscale access
npx paperclipai onboard --yes --bind tailnet
```

## Development Setup

For exploring the source code or contributing:

```bash
git clone https://github.com/paperclipai/paperclip.git
cd paperclip
pnpm install
pnpm dev
```

### Development Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Full dev mode with file watching |
| `pnpm dev:once` | Dev mode without watch |
| `pnpm dev:server` | Server-only mode |
| `pnpm build` | Build all components |
| `pnpm typecheck` | TypeScript type validation |
| `pnpm test:run` | Run tests |
| `pnpm db:generate` | Create database migrations |
| `pnpm db:migrate` | Apply pending migrations |

## Project Structure

```
paperclip/
├── server/      — Backend (Node.js)
├── ui/          — Frontend (React)
├── cli/         — CLI tool (npx paperclipai)
├── packages/    — Shared packages (pnpm monorepo)
├── skills/      — Runtime skill definitions for agents
├── docs/        — Documentation
├── evals/       — Evaluation harnesses
├── tests/       — Test suite
├── docker/      — Docker configuration
└── .agents/     — Agent configuration
```

## Docker Setup

A Dockerfile is included for containerized deployment:

```bash
cd paperclip
docker build -t paperclip .
docker run -p 3100:3100 paperclip
```

## Telemetry

Anonymous usage telemetry is enabled by default. Disable with any of:

```bash
# Environment variable
export PAPERCLIP_TELEMETRY_DISABLED=1

# Or
export DO_NOT_TRACK=1
```

Or set `telemetry.enabled: false` in config. Telemetry is automatically disabled in CI environments.

## Key Concepts

### Agents

Paperclip doesn't build agents — it **manages** them. You bring agents from any provider (Claude Code, Codex, Cursor, custom HTTP agents, Bash scripts) and Paperclip assigns them roles, budgets, and goals.

### Companies

A "company" in Paperclip is an organizational unit with:
- A **mission** (top-level goal)
- An **org chart** (roles and reporting lines)
- **Agents** assigned to roles
- **Budgets** controlling spend per agent

### Heartbeats

Agents run on a heartbeat schedule — they wake up periodically, check for work, execute tasks, and go back to sleep. State persists between heartbeats so agents maintain context.

### Governance

All changes go through approval gates when configured. Decisions are logged immutably for audit purposes. Rollback is supported for configuration changes.

## Next Steps

1. Run `npx paperclipai onboard --yes` to start locally
2. Open **http://localhost:3100** in your browser
3. Create your first company and define its mission
4. Add agents and assign them roles
5. Set budgets and heartbeat schedules
6. Watch your AI company operate
