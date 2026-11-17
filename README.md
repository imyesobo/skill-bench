# skill-bench

Benchmark several Copilot CLI skills that target the same task (e.g. Confluence
extraction) across several models, and capture quality, time, tokens and AI credits.

## Why it's safe to compare skills
Each run of a (skill, model, task, repeat) combination gets a brand-new temporary `HOME`, `COPILOT_HOME` and working directory
that contains **only the skill under test** (as a project skill). Your real
`~/.copilot` (skills, MCP servers, instructions, memories, plugins) and
`~/.agents/skills` are never loaded. Each run is a new session, so AIC starts at 0.
The harness also checks which skill the agent actually called and flags any run where the target skill
wasn't used or a different skill was. Built-in CLI skills can't be removed.

## Why it's fast
Because runs are isolated, they run in parallel (`jobs`). Install and uninstall
happen automatically (copy the skill folder, then delete the temp dir). The harness itself uses no AI credits.

## Usage
```bash
cp bench.example.toml bench.toml   # define skills, models, tasks
./bench.py run bench.toml --dry-run                 # show the run matrix
./bench.py run bench.toml -j 8                      # run everything
./bench.py run bench.toml --only-skill conf-a --only-model claude-sonnet-5
./bench.py report results/<timestamp>               # rebuild summary after manual scoring
./bench.py run examples/demo.toml                   # ~10s smoke test with dummy skills
```
Auth uses `COPILOT_GITHUB_TOKEN`/`GH_TOKEN`, or falls back to `gh auth token`. Skill credentials
come in through `pass_env` and `home_files`.

## Output (`results/<timestamp>/`)
- `summary.md`: per skill×model averages (skill-used rate, regex pass rate, manual
  score, wall time, AIC, premium requests, model calls, tokens, tool calls) plus a per-run table and warnings.
- `results.csv`: one row per run. Fill `manual_score` and `notes`, then run `report` again.
- `runs/<id>/`: `answer.md`, `events.jsonl` (full event stream), `usage.json`
  (CLI usage stats), `stderr.txt`, `command.txt`, `metrics.json`.

Tips: set `baseline = true` to see what the model does without any skill.
Use `repeats = 2-3` to reduce noise. Set `disable_builtin_mcps = true` if the agent
uses the GitHub MCP server instead of your skill.
