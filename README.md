# skill-bench

Benchmark [Agent Skills](https://agentskills.io) (`SKILL.md` folders) that target the same job, for example
several Confluence-extraction skills. You can compare them across **agent harnesses** (Copilot CLI,
Claude Code, OpenCode, or any CLI) and **models**, and the benchmark records answer quality, time,
tokens and cost for each run.

## Why comparisons are fair
Every run gets a fresh temporary `HOME` (with XDG directories) and a working folder. That folder contains
**only the skill under test**, placed where that harness looks for project skills.
Your personal agent setup (installed skills, MCP servers, instructions, memories, plugins) is
never loaded. Only the auth files and env vars you list are copied in. The harness's own
session log is checked to see which skill was actually used. A run is flagged if the target skill
wasn't used or another skill was. Built-in skills that ship with a harness can't be removed.

## Why it's fast and cheap
Isolated runs execute in parallel (`jobs`). Skills are installed and removed automatically:
the folder is copied in and the sandbox is deleted afterwards. The benchmark itself makes no LLM calls.

## Supported harnesses
| harness | skills placed in | cost unit | tested |
|---|---|---|---|
| `copilot` (GitHub Copilot CLI) | `.github/skills/` | AI credits (AIC) | ✅ |
| `opencode` | `.agents/skills/` | USD (OpenCode's list-price estimate) | ✅ |
| `claude-code` | `.claude/skills/` | USD | install + error handling ✅, live run not yet |
| `generic` (any CLI, defined in TOML) | configurable | – (time + checks only) | ✅ |

Each harness reports cost in its own unit, so cost is only comparable between agents that use the same harness.
To add a native adapter for another CLI, subclass `Harness` in `harnesses.py`. It needs four things:
`skills_dir`, `prepare` (copying auth/env into the sandbox), `command`, and `parse`.

## Usage
```bash
cp bench.example.toml bench.toml   # define skills, agents (harness + model), tasks
./bench.py harnesses                         # list adapters
./bench.py run bench.toml --dry-run          # show the run matrix and commands
./bench.py run bench.toml -j 8               # run everything
./bench.py run bench.toml --only-skill conf-a --only-agent claude-opus
./bench.py report results/<timestamp>        # rebuild summary after manual scoring
./bench.py run examples/demo.toml            # ~10s smoke test with dummy skills
```
The run matrix is skills × agents × tasks × repeats. Set `baseline = true` to also run each task with no skill installed.
Requires Python 3.11+ and no extra packages.

## Output (`results/<timestamp>/`)
- `summary.md`: averages for each skill × agent: how often the skill was used, regex pass rate,
  manual score, wall time, cost, model calls, tokens and tool calls. Also includes a per-run table and warnings.
- `results.csv`: one row per run. Fill in `manual_score` and `notes`, then run `report` again.
- `runs/<id>/`: `answer.md`, the harness's raw `stdout.txt` and `stderr.txt`, `command.json`, `metrics.json`,
  and harness extras such as Copilot's `usage.json`.
