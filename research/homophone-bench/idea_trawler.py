"""Trawl ALL of Lingua Weaver for ideas: scan every module's docstring across the
current tree AND every git branch (and notebook markdown), collect the techniques,
and (optionally) ask Nemotron which underused ideas are worth experimenting with.

Read-only: it never edits code. Output -> IDEAS.md.

Run: python idea_trawler.py
"""
from __future__ import annotations

import ast
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                      capture_output=True, text=True).stdout.strip()


def first_doc(src: str) -> str:
    try:
        d = ast.get_docstring(ast.parse(src)) or ""
    except Exception:
        d = ""
    return " ".join(d.strip().split())[:160]


def branches():
    out = subprocess.run(["git", "branch", "-a", "--format=%(refname:short)"],
                         cwd=REPO, capture_output=True, text=True).stdout.split()
    seen, br = set(), []
    for b in out:
        name = b.replace("remotes/origin/", "")
        if name and name != "HEAD" and name not in seen:
            seen.add(name); br.append(b)
    return br


def files_on(branch):
    r = subprocess.run(["git", "ls-tree", "-r", "--name-only", branch],
                       cwd=REPO, capture_output=True, text=True)
    return [f for f in r.stdout.splitlines() if f.endswith(".py")]


def show(branch, path):
    r = subprocess.run(["git", "show", f"{branch}:{path}"], cwd=REPO,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def main():
    cur_files = set(files_on("HEAD"))
    catalogue = {}                       # path -> (branch, docstring)
    # current tree first
    for p in sorted(cur_files):
        catalogue[p] = ("HEAD", first_doc(show("HEAD", p)))
    # then ideas that live ONLY on other branches (not in our tree)
    extra = {}
    for b in branches():
        if b in ("HEAD",):
            continue
        for p in files_on(b):
            if p not in cur_files and p not in extra:
                doc = first_doc(show(b, p))
                if doc:
                    extra[p] = (b, doc)

    lines = ["# Lingua Weaver idea trawl\n",
             f"_{len(catalogue)} modules in tree, {len(extra)} more on other "
             "branches. Read-only scan; originals untouched._\n",
             "\n## Modules in the current tree\n"]
    for p, (_, d) in sorted(catalogue.items()):
        if d:
            lines.append(f"- `{os.path.basename(p)}` — {d}")
    if extra:
        lines.append("\n## Ideas living only on OTHER branches (worth porting)\n")
        for p, (b, d) in sorted(extra.items()):
            lines.append(f"- `{p}` @ `{b}` — {d}")

    # optional: Nemotron picks underused ideas to experiment with
    try:
        import os as _os
        import urllib.request
        import _load_env; _load_env.load_keys()
        key = _os.environ.get("OPENROUTER_API_KEY")
        if key:
            summary = "\n".join(lines[3:])[:6000]
            prompt = ("Here are module summaries from a homophonic EN<->FR "
                      "translation project. List the 8 most UNDERUSED or promising "
                      "ideas to experiment with next for better sound+meaning "
                      "quality, each one line with why. Be specific.\n\n" + summary)
            body = json.dumps({"model": "nvidia/nemotron-3-super-120b-a12b:free",
                               "messages": [{"role": "user", "content": prompt}],
                               "max_tokens": 900, "reasoning": {"enabled": False}}).encode()
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                                         data=body, headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
            lines.append("\n## Nemotron: underused ideas to experiment with next\n")
            lines.append(txt)
    except Exception as e:
        lines.append(f"\n(LLM synthesis skipped: {e})")

    open(os.path.join(ROOT, "IDEAS.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"scanned {len(catalogue)} tree modules + {len(extra)} cross-branch "
          f"ideas -> IDEAS.md")


if __name__ == "__main__":
    main()
