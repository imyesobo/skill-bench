"""Harness adapters.

A harness adapter knows how to, for one agent CLI:
  * where project skills must be placed (``skills_dir``),
  * how to isolate its config/auth inside a throw-away HOME (``prepare``),
  * how to build a headless command (``command``),
  * how to turn its output into normalized metrics (``parse``).

To support a new agent CLI, subclass ``Harness`` and register it in ``HARNESSES``,
or use the config-only ``generic`` harness.
"""
import json
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


def empty_metrics():
    return {
        "answer": "",
        "exit_code": None,
        "skills_used": [],
        "tools": {},
        "subagents": 0,
        "tool_errors": 0,
        "model_calls": 0,
        "tok_input": 0,
        "tok_cache_read": 0,
        "tok_cache_write": 0,
        "tok_output": 0,
        "tok_reasoning": 0,
        "cost": None,
        "cost_unit": "",
        "extra": {},
    }


def read_jsonl(path: Path):
    if not path.exists():
        return
    for line in path.open(encoding="utf-8", errors="replace"):
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def copy_from_home(rel: str, home: Path):
    src, dst = Path.home() / rel, home / rel
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


SKILL_PATH_RE = re.compile(r"skills/([^/\"'\s]+)/SKILL\.md")


def skills_from_reads(args) -> list:
    """Fallback detection: the agent read <skills_dir>/<dir>/SKILL.md directly."""
    return SKILL_PATH_RE.findall(json.dumps(args)) if args else []


@lru_cache(maxsize=1)
def gh_token() -> str:
    token = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True).stdout.strip()
    return token or ""


class Harness:
    name = "base"
    skills_dir = ".agents/skills"
    auth_home_files: list = []   # copied from the real HOME into the sandbox HOME
    auth_env: list = []          # env vars passed through when set

    def __init__(self, agent: dict):
        self.agent = agent
        self.skills_dir = agent.get("skills_dir", self.skills_dir)

    def prepare(self, home: Path, work: Path, env: dict):
        for rel in self.auth_home_files:
            copy_from_home(rel, home)
        for k in self.auth_env:
            if k in os.environ:
                env[k] = os.environ[k]

    def command(self, prompt: str, out: Path) -> list:
        raise NotImplementedError

    def parse(self, out: Path) -> dict:
        raise NotImplementedError


class Copilot(Harness):
    """GitHub Copilot CLI (`copilot -p`). Cost unit: AI credits (AIC)."""
    name = "copilot"
    skills_dir = ".github/skills"

    def prepare(self, home, work, env):
        super().prepare(home, work, env)
        token = gh_token()
        if token:
            env["COPILOT_GITHUB_TOKEN"] = token
        env.update(
            COPILOT_HOME=str(home / ".copilot"),
            COPILOT_CACHE_HOME=os.environ.get("COPILOT_CACHE_HOME", str(Path.home() / ".cache" / "copilot")),
            COPILOT_AUTO_UPDATE="false",
            COPILOT_DISABLE_LOGIN_SHELL_ENV="1",
        )

    def command(self, prompt, out):
        a = self.agent
        cmd = ["copilot", "-p", prompt, "--output-format", "json", "--usage-output-file", str(out / "usage.json"),
               "--no-ask-user", "--no-custom-instructions", "--allow-all", "--no-auto-update"]
        if a.get("model"):
            cmd += ["--model", a["model"]]
        if a.get("effort"):
            cmd += ["--reasoning-effort", a["effort"]]
        if a.get("max_cost"):
            cmd += ["--max-ai-credits", str(a["max_cost"])]
        return cmd

    def parse(self, out):
        m = empty_metrics()
        m["cost_unit"] = "AIC"
        for e in read_jsonl(out / "stdout.txt"):
            t, d = e.get("type"), e.get("data", {})
            if t == "tool.execution_start":
                name, args = d.get("toolName", "?"), d.get("arguments", {})
                m["tools"][name] = m["tools"].get(name, 0) + 1
                if name == "skill":
                    m["skills_used"].append(args.get("skill", "?"))
                else:
                    m["skills_used"] += skills_from_reads(args)
                m["subagents"] += name == "task"
            elif t == "tool.execution_complete" and not d.get("success", True):
                m["tool_errors"] += 1
            elif t == "assistant.message" and d.get("content") and not d.get("parentToolCallId"):
                m["answer"] = d["content"]
            elif t == "result":
                m["exit_code"] = e.get("exitCode")
        usage_path = out / "usage.json"
        if usage_path.exists():
            u = json.loads(usage_path.read_text())
            td = u.get("tokenDetails", {})
            mm = list(u.get("modelMetrics", {}).values())
            m.update(
                tok_input=td.get("input", {}).get("tokenCount", 0),
                tok_cache_read=td.get("cache_read", {}).get("tokenCount", 0),
                tok_cache_write=td.get("cache_write", {}).get("tokenCount", 0),
                tok_output=td.get("output", {}).get("tokenCount", 0),
                tok_reasoning=sum(x.get("usage", {}).get("reasoningTokens", 0) for x in mm),
                model_calls=sum(x.get("requests", {}).get("count", 0) for x in mm),
                cost=u.get("totalNanoAiu", 0) / 1e9,
            )
            m["extra"] = {"premium_requests": u.get("totalPremiumRequestCost"),
                          "cost_status": u.get("aiCreditsStatus")}
        return m


class ClaudeCode(Harness):
    """Anthropic Claude Code (`claude -p`). Cost unit: USD as reported by Claude Code."""
    name = "claude-code"
    skills_dir = ".claude/skills"
    auth_home_files = [".claude/.credentials.json"]
    auth_env = ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "AWS_REGION", "AWS_PROFILE",
                "ANTHROPIC_VERTEX_PROJECT_ID", "CLOUD_ML_REGION"]

    def prepare(self, home, work, env):
        super().prepare(home, work, env)
        env["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
        env["DISABLE_AUTOUPDATER"] = "1"

    def command(self, prompt, out):
        a = self.agent
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--dangerously-skip-permissions", "--strict-mcp-config"]
        if a.get("model"):
            cmd += ["--model", a["model"]]
        if a.get("effort"):
            cmd += ["--effort", a["effort"]]
        if a.get("max_cost"):
            cmd += ["--max-budget-usd", str(a["max_cost"])]
        return cmd

    def parse(self, out):
        m = empty_metrics()
        m["cost_unit"] = "USD"
        for e in read_jsonl(out / "stdout.txt"):
            t = e.get("type")
            if t == "assistant":
                if e.get("parent_tool_use_id"):
                    continue
                m["model_calls"] += e.get("message", {}).get("model") != "<synthetic>"
                for c in e.get("message", {}).get("content", []):
                    if c.get("type") == "tool_use":
                        name, args = c.get("name", "?"), c.get("input", {})
                        m["tools"][name] = m["tools"].get(name, 0) + 1
                        if name == "Skill":
                            m["skills_used"].append(args.get("skill") or args.get("command") or "?")
                        else:
                            m["skills_used"] += skills_from_reads(args)
                        m["subagents"] += name in ("Task", "Agent")
            elif t == "user":
                for c in e.get("message", {}).get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("is_error"):
                        m["tool_errors"] += 1
            elif t == "result":
                u = e.get("usage", {})
                m.update(
                    answer=e.get("result") or "",
                    exit_code=1 if e.get("is_error") else 0,
                    cost=e.get("total_cost_usd"),
                    tok_input=u.get("input_tokens", 0),
                    tok_cache_read=u.get("cache_read_input_tokens", 0),
                    tok_cache_write=u.get("cache_creation_input_tokens", 0),
                    tok_output=u.get("output_tokens", 0),
                )
                m["extra"] = {"num_turns": e.get("num_turns"), "subtype": e.get("subtype")}
        return m


class OpenCode(Harness):
    """sst OpenCode (`opencode run`). Cost unit: USD as reported by OpenCode (list prices)."""
    name = "opencode"
    skills_dir = ".agents/skills"
    auth_home_files = [".local/share/opencode/auth.json"]
    auth_env = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"]

    def prepare(self, home, work, env):
        super().prepare(home, work, env)
        # Share the package cache (plugins/provider SDKs) so runs don't re-download them.
        env["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"

    def command(self, prompt, out):
        a = self.agent
        cmd = ["opencode", "run", "--format", "json", "--auto"]
        if a.get("model"):
            cmd += ["--model", a["model"]]
        if a.get("effort"):
            cmd += ["--variant", a["effort"]]
        return cmd + [prompt]

    def parse(self, out):
        m = empty_metrics()
        m["cost_unit"] = "USD"
        cost, step_text = 0.0, []
        for e in read_jsonl(out / "stdout.txt"):
            t, p = e.get("type"), e.get("part", {})
            if t == "step_start":
                step_text = []
            elif t == "text":
                step_text.append(p.get("text", ""))
            elif t == "tool_use":
                name, st = p.get("tool", "?"), p.get("state", {})
                args = st.get("input", {})
                m["tools"][name] = m["tools"].get(name, 0) + 1
                if name == "skill":
                    m["skills_used"].append(args.get("name", "?"))
                else:
                    m["skills_used"] += skills_from_reads(args)
                m["subagents"] += name == "task"
                m["tool_errors"] += st.get("status") == "error"
            elif t == "step_finish":
                tk = p.get("tokens", {})
                m["model_calls"] += 1
                m["tok_input"] += tk.get("input", 0)
                m["tok_output"] += tk.get("output", 0)
                m["tok_reasoning"] += tk.get("reasoning", 0)
                m["tok_cache_read"] += tk.get("cache", {}).get("read", 0)
                m["tok_cache_write"] += tk.get("cache", {}).get("write", 0)
                cost += p.get("cost", 0) or 0
                if p.get("reason") == "stop":
                    m["answer"] = "".join(step_text).strip()
            elif t == "error":
                m["exit_code"] = 1
        m["cost"] = cost
        return m


class Generic(Harness):
    """Any other agent CLI, configured entirely from bench.toml.

    command = ["mycli", "--model", "{model}", "-p", "{prompt}"]   # placeholders: prompt model effort workdir outdir
    skills_dir = ".agents/skills"
    The agent's stdout is taken as the answer; only wall time and checks are measured.
    """
    name = "generic"

    def __init__(self, agent):
        super().__init__(agent)
        if "command" not in agent:
            raise ValueError(f"agent '{agent['id']}': harness 'generic' needs a 'command' list")
        self.auth_home_files = agent.get("auth_home_files", [])
        self.auth_env = agent.get("auth_env", [])

    def command(self, prompt, out):
        vals = {"prompt": prompt, "model": self.agent.get("model", ""), "effort": self.agent.get("effort", ""),
                "outdir": str(out), "workdir": "."}
        return [str(x).format(**vals) for x in self.agent["command"]]

    def parse(self, out):
        m = empty_metrics()
        text = (out / "stdout.txt").read_text(errors="replace") if (out / "stdout.txt").exists() else ""
        m["answer"] = text.strip()
        m["skills_used"] = skills_from_reads(text)
        return m


HARNESSES = {h.name: h for h in (Copilot, ClaudeCode, OpenCode, Generic)}


def get_harness(agent: dict) -> Harness:
    name = agent.get("harness")
    if name not in HARNESSES:
        raise ValueError(f"agent '{agent.get('id')}': unknown harness '{name}'. Available: {', '.join(HARNESSES)}")
    return HARNESSES[name](agent)
