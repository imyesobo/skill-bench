#!/usr/bin/env python3
"""skill-bench: compare Agent Skills (SKILL.md) across agent harnesses and models.

Each (skill, agent, task, repeat) run gets its own throw-away HOME and working
directory containing ONLY the skill under test, so skills never interfere with
each other or with your personal configuration. Isolated runs execute in
parallel. The harness itself makes no LLM calls.

Usage:
  ./bench.py run bench.toml [-j N] [--only-skill X] [--only-agent A] [--only-task T] [--dry-run]
  ./bench.py report results/<timestamp>
  ./bench.py harnesses
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

from harnesses import HARNESSES, copy_from_home, get_harness

HERE = Path(__file__).resolve().parent
PRINT_LOCK = threading.Lock()
BASE_ENV = ("PATH", "LANG", "LC_ALL", "TERM", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")


def log(msg):
    with PRINT_LOCK:
        print(msg, flush=True)


def skill_name(skill_dir: Path) -> str:
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^name:\s*['\"]?([^'\"\n]+)", text, re.M)
    return m.group(1).strip() if m else skill_dir.name


def resolve_skill(spec: dict, base: Path, cache: Path) -> Path:
    """Return a local directory containing SKILL.md (git sources are cloned once)."""
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
        if not args.only_skill or s["id"] in args.only_skill:
            p = resolve_skill(s, base, cache)
            skills.append({**s, "dir": p, "name": skill_name(p)})
    if st.get("baseline") and (not args.only_skill or "baseline" in args.only_skill):
        skills.append({"id": "baseline", "dir": None, "name": None})

    agents = [a for a in cfg.get("agents", []) if not args.only_agent or a["id"] in args.only_agent]
    for a in agents:
        get_harness(a)  # validate early
    tasks = [t for t in cfg.get("tasks", []) if not args.only_task or t["id"] in args.only_task]
    if not (skills and agents and tasks):
        sys.exit("Nothing to run: need at least one skill (or baseline), one agent and one task.")
    return [
        {"skill": s, "agent": a, "task": t, "rep": r}
        for s in skills for a in agents for t in tasks for r in range(1, st.get("repeats", 1) + 1)
    ]


def make_env(st: dict, agent: dict, home: Path):
    env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
    env.update(
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_STATE_HOME=str(home / ".local" / "state"),
        XDG_CACHE_HOME=str(home / ".cache"),
    )
    for k in st.get("pass_env", []) + agent.get("pass_env", []):
        if k in os.environ:
            env[k] = os.environ[k]
    for rel in st.get("home_files", []) + agent.get("home_files", []):
        copy_from_home(rel, home)
    return env


def check_answer(task: dict, answer: str):
    checks, ok = [], True
    for pat in task.get("must_contain", []):
        hit = re.search(pat, answer, re.I | re.S) is not None
        checks.append(("+" if hit else "-") + pat)
        ok &= hit
    for pat in task.get("must_not_contain", []):
        hit = re.search(pat, answer, re.I | re.S) is not None
        checks.append(("-" if hit else "+") + "!" + pat)
        ok &= not hit
    return " ".join(checks), (ok if checks else "")


def run_one(cfg, base: Path, run, out_dir: Path, dry: bool):
    st = cfg.get("settings", {})
    s, a, t = run["skill"], run["agent"], run["task"]
    h = get_harness(a)
    rid = f"{s['id']}__{a['id']}__{t['id']}__r{run['rep']}"
    rdir = out_dir / "runs" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    cmd = h.command(t["prompt"], rdir) + a.get("args", [])
    (rdir / "command.json").write_text(json.dumps(cmd))
    if dry:
        return

    sandbox = Path(tempfile.mkdtemp(prefix="skillbench-"))
    home, work = sandbox / "home", sandbox / "work"
    home.mkdir()
    work.mkdir()
    if st.get("fixtures_dir"):
        shutil.copytree(base / st["fixtures_dir"], work, dirs_exist_ok=True)
    if s["dir"]:
        shutil.copytree(s["dir"], work / h.skills_dir / s["dir"].name, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    env = make_env(st, a, home)
    h.prepare(home, work, env)
    env.update({k: os.path.expandvars(str(v)) for k, v in {**st.get("env", {}), **a.get("env", {})}.items()})

    timeout = t.get("timeout", st.get("timeout", 600))
    t0 = time.monotonic()
    timed_out, rc = False, None
    with (rdir / "stdout.txt").open("w") as fo, (rdir / "stderr.txt").open("w") as fe:
        try:
            rc = subprocess.run(cmd, cwd=work, env=env, stdout=fo, stderr=fe,
                                stdin=subprocess.DEVNULL, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except FileNotFoundError:
            fe.write(f"command not found: {cmd[0]}\n")
            rc = 127
    wall = round(time.monotonic() - t0, 2)
    if st.get("keep_sandbox"):
        (rdir / "sandbox.txt").write_text(str(sandbox))
    else:
        shutil.rmtree(sandbox, ignore_errors=True)

    m = h.parse(rdir)
    (rdir / "answer.md").write_text(m["answer"], encoding="utf-8")
    targets = {s["name"], s["dir"].name} if s["dir"] else set()
    checks, passed = check_answer(t, m["answer"])
    row = {
        "run_id": rid, "skill": s["id"], "skill_name": s["name"] or "",
        "agent": a["id"], "harness": h.name, "model": a.get("model", ""), "effort": a.get("effort", ""),
        "task": t["id"], "rep": run["rep"],
        "wall_s": wall, "timed_out": timed_out,
        "exit_code": rc if m["exit_code"] is None or rc not in (0, None) else m["exit_code"],
        "cost": round(m["cost"], 6) if m["cost"] is not None else "", "cost_unit": m["cost_unit"],
        "model_calls": m["model_calls"],
        "tok_input": m["tok_input"], "tok_cache_read": m["tok_cache_read"],
        "tok_cache_write": m["tok_cache_write"], "tok_output": m["tok_output"], "tok_reasoning": m["tok_reasoning"],
        "tool_calls": sum(m["tools"].values()),
        "tools": " ".join(f"{k}:{v}" for k, v in sorted(m["tools"].items())),
        "subagents": m["subagents"], "tool_errors": m["tool_errors"],
        "skills_used": " ".join(dict.fromkeys(m["skills_used"])),
        "target_skill_used": bool(targets & set(m["skills_used"])) if targets else "",
        "foreign_skill_used": bool(set(m["skills_used"]) - targets),
        "checks": checks, "checks_pass": passed,
        "answer_chars": len(m["answer"]),
        "extra": json.dumps(m["extra"]) if m["extra"] else "",
        "manual_score": "", "notes": "",
    }
    (rdir / "metrics.json").write_text(json.dumps(row, indent=2))
    cost = f"{row['cost']:.4g} {row['cost_unit']}" if row["cost"] != "" else "cost n/a"
    flag = "" if row["target_skill_used"] in (True, "") else "  ⚠ target skill NOT used"
    log(f"✓ {rid}: {wall}s, {cost}, checks={passed}{flag}")


def report(out_dir: Path):
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("runs/*/metrics.json"))]
    if not rows:
        sys.exit(f"No results in {out_dir}")
    csv_path = out_dir / "results.csv"
    if csv_path.exists():  # keep manual scores/notes typed into the CSV
        prev = {r["run_id"]: r for r in csv.DictReader(csv_path.open())}
        for r in rows:
            p = prev.get(r["run_id"], {})
            r["manual_score"], r["notes"] = p.get("manual_score", ""), p.get("notes", "")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(xs):
        xs = [float(x) for x in xs if x != ""]
        return sum(xs) / len(xs) if xs else None

    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    def frac(rs, pred):
        return f"{sum(1 for r in rs if pred(r))}/{len(rs)}" if rs else "-"

    groups = {}
    for r in rows:
        groups.setdefault((r["skill"], r["agent"]), []).append(r)
    lines = [
        f"# Skill benchmark — {out_dir.name}", "",
        "Averages per run. `skill used` = target skill invoked; `pass` = automatic regex checks; "
        "`manual` = average of the manual_score column in results.csv. "
        "Cost is reported by each harness in its own unit and is only comparable within the same unit.", "",
        "| skill | agent | harness | model | runs | skill used | pass | manual | wall s | cost | calls | in tok | out tok | tools |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (sk, ag), rs in sorted(groups.items(), key=lambda kv: (kv[1][0]["cost_unit"], avg(r["cost"] for r in kv[1]) or 0)):
        chk = [r for r in rs if r["checks_pass"] != ""]
        cost = avg(r["cost"] for r in rs)
        lines.append("| " + " | ".join([
            sk, ag, rs[0]["harness"], rs[0]["model"] or "-", str(len(rs)),
            "-" if sk == "baseline" else frac(rs, lambda r: r["target_skill_used"] is True),
            frac(chk, lambda r: r["checks_pass"] is True),
            fmt(avg(r["manual_score"] for r in rs), ".2f"),
            fmt(avg(r["wall_s"] for r in rs), ".1f"),
            "-" if cost is None else f"{cost:.4g} {rs[0]['cost_unit']}",
            fmt(avg(r["model_calls"] for r in rs), ".1f"),
            fmt(avg(r["tok_input"] + r["tok_cache_read"] + r["tok_cache_write"] for r in rs), ".0f"),
            fmt(avg(r["tok_output"] for r in rs), ".0f"),
            fmt(avg(r["tool_calls"] for r in rs), ".1f"),
        ]) + " |")
    lines += ["", "## Per run", "",
              "| task | skill | agent | rep | wall s | cost | skill used | pass | skills invoked |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], r["agent"], r["skill"], r["rep"])):
        cost = f"{r['cost']:.4g} {r['cost_unit']}" if r["cost"] != "" else "-"
        lines.append(f"| {r['task']} | {r['skill']} | {r['agent']} | {r['rep']} | {r['wall_s']} | {cost} | "
                     f"{r['target_skill_used']} | {r['checks_pass']} | {r['skills_used']} |")
    warn = [r["run_id"] for r in rows
            if r["foreign_skill_used"] or r["timed_out"] or r["exit_code"] not in (0, None)
            or (r["skill"] != "baseline" and r["target_skill_used"] is False)]
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
    r.add_argument("--only-agent", nargs="+")
    r.add_argument("--only-task", nargs="+")
    r.add_argument("--dry-run", action="store_true", help="print the run matrix and commands; run nothing")
    r.add_argument("--out", help="results directory (default: results/<timestamp>)")
    rp = sub.add_parser("report", help="rebuild summary.md (e.g. after manual scoring)")
    rp.add_argument("dir")
    sub.add_parser("harnesses", help="list supported harnesses")
    args = ap.parse_args()

    if args.cmd == "report":
        return report(Path(args.dir))
    if args.cmd == "harnesses":
        for name, cls in HARNESSES.items():
            print(f"{name:12} skills_dir={cls.skills_dir:15} {(cls.__doc__ or '').strip().splitlines()[0]}")
        return

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
    if args.dry_run:
        for p in sorted(out_dir.glob("runs/*/command.json")):
            print(f"{p.parent.name}\n    {' '.join(json.loads(p.read_text()))}")
        shutil.rmtree(out_dir)
        return
    log(f"Total wall time: {time.monotonic() - t0:.1f}s")
    report(out_dir)


if __name__ == "__main__":
    main()
