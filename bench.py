#!/usr/bin/env python3
"""Skill benchmark harness for GitHub Copilot CLI.

Each (skill, model, task, repeat) run gets its own throw-away HOME, COPILOT_HOME
and working directory containing ONLY the skill under test, so skills can never
interfere with each other. Because runs are fully isolated they execute in
parallel. The harness itself makes no LLM calls.

Usage:
  ./bench.py run bench.toml [--jobs N] [--only-skill X] [--only-model M] [--only-task T] [--dry-run]
  ./bench.py report results/<timestamp>
"""
import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRINT_LOCK = threading.Lock()
TOKEN_LOCK = threading.Lock()
_TOKEN = None


def log(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def skill_name(skill_dir: Path) -> str:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^name:\s*['\"]?([^'\"\n]+)", text, re.M)
    return m.group(1).strip() if m else skill_dir.name


def resolve_skill(spec: dict, base: Path, cache: Path) -> Path:
    """Return a local directory containing SKILL.md (clones git sources once)."""
    if "git" in spec:
        dest = cache / re.sub(r"[^\w.-]", "_", spec["git"] + spec.get("ref", ""))
        if not dest.exists():
            cmd = ["git", "clone", "--quiet", "--depth", "1"]
            if spec.get("ref"):
                cmd += ["--branch", spec["ref"]]
            subprocess.run(cmd + [spec["git"], str(dest)], check=True)
        path = dest / spec.get("subdir", "")
    else:
        path = (base / os.path.expanduser(spec["path"])).resolve()
    if not (path / "SKILL.md").exists():
        sys.exit(f"Skill '{spec['id']}': no SKILL.md in {path}")
    return path


def build_runs(cfg, base: Path, args):
    cache = HERE / ".skill-cache"
    cache.mkdir(exist_ok=True)
    st = cfg.get("settings", {})
    skills = []
    for s in cfg.get("skills", []):
        if args.only_skill and s["id"] not in args.only_skill:
            continue
        p = resolve_skill(s, base, cache)
        skills.append({**s, "dir": p, "name": skill_name(p)})
    if st.get("baseline") and (not args.only_skill or "baseline" in args.only_skill):
        skills.append({"id": "baseline", "dir": None, "name": None})

    models = []
    for m in cfg["models"]:
        m = {"name": m} if isinstance(m, str) else m
        if not args.only_model or m["name"] in args.only_model:
            models.append(m)
    tasks = [t for t in cfg["tasks"] if not args.only_task or t["id"] in args.only_task]
    repeats = st.get("repeats", 1)
    return [
        {"skill": s, "model": m, "task": t, "rep": r}
        for s in skills for m in models for t in tasks for r in range(1, repeats + 1)
    ]


def github_token():
    global _TOKEN
    with TOKEN_LOCK:
        if _TOKEN is None:
            _TOKEN = os.environ.get("COPILOT_GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=True).stdout.strip()
        return _TOKEN


def make_env(cfg, home: Path):
    st = cfg.get("settings", {})
    real_home = Path.home()
    env = {k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TERM", "TZ", "SSL_CERT_FILE") if k in os.environ}
    for k in st.get("pass_env", []):  # e.g. CONFLUENCE_TOKEN
        if k in os.environ:
            env[k] = os.environ[k]
    env.update({k: os.path.expandvars(str(v)) for k, v in st.get("env", {}).items()})
    env.update(
        HOME=str(home),
        COPILOT_HOME=str(home / ".copilot"),
        COPILOT_CACHE_HOME=os.environ.get("COPILOT_CACHE_HOME", str(real_home / ".cache" / "copilot")),
        COPILOT_GITHUB_TOKEN=github_token(),
        COPILOT_AUTO_UPDATE="false",
        COPILOT_DISABLE_LOGIN_SHELL_ENV="1",
        COPILOT_DISABLE_DESKTOP_NOTIFICATIONS="1",
    )
    # Copy files a skill needs from the real HOME (e.g. .config/confluence, .netrc).
    for rel in st.get("home_files", []):
        src, dst = real_home / rel, home / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return env


def analyse(events_path: Path, usage_path: Path, run):
    skills_used, tools, answer, subagents, errors, result = [], {}, "", 0, 0, {}
    if events_path.exists():
        for line in events_path.open(encoding="utf-8"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            t, d = e.get("type"), e.get("data", {})
            if t == "tool.execution_start":
                name = d.get("toolName", "?")
                tools[name] = tools.get(name, 0) + 1
                if name == "skill":
                    skills_used.append(d.get("arguments", {}).get("skill", "?"))
                elif name == "task":
                    subagents += 1
            elif t == "tool.execution_complete" and not d.get("success", True):
                errors += 1
            elif t == "assistant.message" and d.get("content") and not d.get("parentToolCallId"):
                answer = d["content"]
            elif t == "result":
                result = e
    usage = json.loads(usage_path.read_text()) if usage_path.exists() else {}
    td = usage.get("tokenDetails", {})
    mm = usage.get("modelMetrics", {}).values()

    def tok(k):
        return td.get(k, {}).get("tokenCount", 0)

    task, target = run["task"], run["skill"]["name"]
    checks, ok = [], True
    for pat in task.get("must_contain", []):
        hit = re.search(pat, answer, re.I | re.S) is not None
        checks.append(("+" if hit else "-") + pat)
        ok &= hit
    for pat in task.get("must_not_contain", []):
        hit = re.search(pat, answer, re.I | re.S) is not None
        checks.append(("-" if hit else "+") + "!" + pat)
        ok &= not hit
    return {
        "exit_code": result.get("exitCode"),
        "aic": round(usage.get("totalNanoAiu", 0) / 1e9, 4),
        "aic_status": usage.get("aiCreditsStatus", ""),
        "premium_requests": usage.get("totalPremiumRequestCost", 0),
        "session_s": round(result.get("usage", {}).get("sessionDurationMs", 0) / 1000, 2),
        "api_s": round(usage.get("totalApiDurationMs", 0) / 1000, 2),
        "model_calls": sum(m.get("requests", {}).get("count", 0) for m in mm),
        "tok_input": tok("input"),
        "tok_cache_read": tok("cache_read"),
        "tok_cache_write": tok("cache_write"),
        "tok_output": tok("output"),
        "tok_reasoning": sum(m.get("usage", {}).get("reasoningTokens", 0) for m in mm),
        "tool_calls": sum(tools.values()),
        "tools": " ".join(f"{k}:{v}" for k, v in sorted(tools.items())),
        "subagents": subagents,
        "skills_used": " ".join(skills_used),
        "target_skill_used": (target in skills_used) if target else "",
        "foreign_skill_used": any(s != target for s in skills_used),
        "tool_errors": errors,
        "checks": " ".join(checks),
        "checks_pass": ok if checks else "",
        "answer_chars": len(answer),
    }, answer


def run_one(cfg, base: Path, run, out_dir: Path, dry):
    st = cfg.get("settings", {})
    s, m, t = run["skill"], run["model"], run["task"]
    rid = f"{s['id']}__{m['name']}__{t['id']}__r{run['rep']}"
    rdir = out_dir / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)

    effort = m.get("effort") or st.get("reasoning_effort")
    cmd = [
        "copilot", "-p", t["prompt"], "--model", m["name"],
        "--output-format", "json", "--usage-output-file", str(rdir / "usage.json"),
        "--no-ask-user", "--no-custom-instructions", "--allow-all", "--no-auto-update",
    ]
    if effort:
        cmd += ["--reasoning-effort", effort]
    if st.get("max_ai_credits"):
        cmd += ["--max-ai-credits", str(st["max_ai_credits"])]
    if st.get("disable_builtin_mcps"):
        cmd += ["--disable-builtin-mcps"]
    cmd += st.get("extra_args", [])
    (rdir / "command.txt").write_text(" ".join(cmd))
    if dry:
        return {"run_id": rid}

    sandbox = Path(tempfile.mkdtemp(prefix="skillbench-"))
    home, work = sandbox / "home", sandbox / "work"
    (home / ".copilot").mkdir(parents=True)
    work.mkdir()
    if st.get("fixtures_dir"):
        shutil.copytree(base / st["fixtures_dir"], work, dirs_exist_ok=True)
    if s["dir"]:
        shutil.copytree(s["dir"], work / ".github" / "skills" / s["dir"].name, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    env = make_env(cfg, home)

    timeout = t.get("timeout", st.get("timeout", 600))
    t0 = time.monotonic()
    timed_out = False
    with (rdir / "events.jsonl").open("w") as fo, (rdir / "stderr.txt").open("w") as fe:
        try:
            subprocess.run(cmd, cwd=work, env=env, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
    wall = round(time.monotonic() - t0, 2)
    metrics, answer = analyse(rdir / "events.jsonl", rdir / "usage.json", run)
    (rdir / "answer.md").write_text(answer, encoding="utf-8")
    if not st.get("keep_sandbox"):
        shutil.rmtree(sandbox, ignore_errors=True)
    else:
        (rdir / "sandbox.txt").write_text(str(sandbox))
    row = {
        "run_id": rid, "skill": s["id"], "skill_name": s["name"] or "", "model": m["name"],
        "effort": effort or "", "task": t["id"], "rep": run["rep"], "wall_s": wall, "timed_out": timed_out,
        **metrics, "manual_score": "", "notes": "",
    }
    (rdir / "metrics.json").write_text(json.dumps(row, indent=2))
    flag = "" if metrics["target_skill_used"] in (True, "") else "  ⚠ target skill NOT used"
    log(f"✓ {rid}: {wall}s, {metrics['aic']} AIC, checks={metrics['checks_pass']}{flag}")
    return row


def report(out_dir: Path):
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("runs/*/metrics.json"))]
    if not rows:
        sys.exit(f"No results in {out_dir}")
    # Preserve manual scores/notes already typed into results.csv.
    csv_path = out_dir / "results.csv"
    if csv_path.exists():
        prev = {r["run_id"]: r for r in csv.DictReader(csv_path.open())}
        for r in rows:
            p = prev.get(r["run_id"], {})
            r["manual_score"], r["notes"] = p.get("manual_score", ""), p.get("notes", "")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0

    def frac(rs, pred):
        return f"{sum(1 for r in rs if pred(r))}/{len(rs)}" if rs else "-"

    groups = {}
    for r in rows:
        groups.setdefault((r["skill"], r["model"]), []).append(r)
    lines = [
        f"# Skill benchmark — {out_dir.name}", "",
        "Averages per run. `skill used` = target skill invoked; `pass` = automatic regex checks; "
        "`manual` = average of your manual_score column in results.csv.", "",
        "| skill | model | runs | skill used | pass | manual | wall s | AIC | prem req | calls | in tok | out tok | tools |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (sk, mo), rs in sorted(groups.items(), key=lambda kv: avg(r["aic"] for r in kv[1])):
        chk = [r for r in rs if r["checks_pass"] != ""]
        manual = [float(r["manual_score"]) for r in rs if str(r["manual_score"]).strip()]
        lines.append("| " + " | ".join([
            sk, mo, str(len(rs)),
            "-" if sk == "baseline" else frac(rs, lambda r: r["target_skill_used"] is True),
            frac(chk, lambda r: r["checks_pass"] is True),
            f"{avg(manual):.2f}" if manual else "-",
            f"{avg(r['wall_s'] for r in rs):.1f}",
            f"{avg(r['aic'] for r in rs):.3f}",
            f"{avg(r['premium_requests'] for r in rs):.2f}",
            f"{avg(r['model_calls'] for r in rs):.1f}",
            f"{avg(r['tok_input'] + r['tok_cache_read'] + r['tok_cache_write'] for r in rs):.0f}",
            f"{avg(r['tok_output'] for r in rs):.0f}",
            f"{avg(r['tool_calls'] for r in rs):.1f}",
        ]) + " |")
    lines += ["", "## Per run", "",
              "| task | skill | model | rep | wall s | AIC | skill used | pass | skills invoked |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], r["model"], r["aic"])):
        lines.append(f"| {r['task']} | {r['skill']} | {r['model']} | {r['rep']} | {r['wall_s']} | {r['aic']} | "
                     f"{r['target_skill_used']} | {r['checks_pass']} | {r['skills_used']} |")
    warn = [r["run_id"] for r in rows
            if r["foreign_skill_used"] or r["timed_out"] or r["exit_code"] not in (0, None)
            or (r["skill"] != "baseline" and r["target_skill_used"] is not True)]
    if warn:
        lines += ["", "## ⚠ Runs needing attention (skill not used / foreign skill / timeout / error)", ""]
        lines += [f"- {w}" for w in warn]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print((out_dir / "summary.md").read_text())
    print(f"Answers: {out_dir}/runs/*/answer.md")
    print(f"CSV (fill manual_score/notes, then `./bench.py report {out_dir}`): {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="execute a benchmark")
    r.add_argument("config")
    r.add_argument("--jobs", "-j", type=int, help="parallel runs (default: settings.jobs or 4)")
    r.add_argument("--only-skill", nargs="+")
    r.add_argument("--only-model", nargs="+")
    r.add_argument("--only-task", nargs="+")
    r.add_argument("--dry-run", action="store_true", help="print the run matrix without calling Copilot")
    r.add_argument("--out", help="results directory (default: results/<timestamp>)")
    rp = sub.add_parser("report", help="rebuild summary.md (e.g. after manual scoring)")
    rp.add_argument("dir")
    args = ap.parse_args()

    if args.cmd == "report":
        return report(Path(args.dir))

    cfg_path = Path(args.config).resolve()
    cfg = tomllib.loads(cfg_path.read_text())
    runs = build_runs(cfg, cfg_path.parent, args)
    out_dir = Path(args.out) if args.out else HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_path, out_dir / "bench.toml")
    jobs = args.jobs or cfg.get("settings", {}).get("jobs", 4)
    log(f"{len(runs)} runs, {jobs} in parallel → {out_dir}")
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(run_one, cfg, cfg_path.parent, run, out_dir, args.dry_run) for run in runs]
        for f in cf.as_completed(futs):
            if f.exception():
                log(f"✗ run failed: {f.exception()!r}")
    log(f"Total wall time: {time.monotonic() - t0:.1f}s")
    if args.dry_run:
        for p in sorted(out_dir.glob("runs/*/command.txt")):
            print(p.parent.name)
        shutil.rmtree(out_dir)
        return
    report(out_dir)


if __name__ == "__main__":
    main()
