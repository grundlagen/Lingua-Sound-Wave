"""chain_engine.py — the engine routed through the breakthrough: the woven
chain-web, used to its furthest-developed form.

Every content word transfers to the target through a real ALTERNATION CHAIN
(interleaved sound/meaning hops) drawn from the precomputed, looped web —
not a one-shot decode. Priority of evidence, best first:
  1. loop-certified pairs   (loop-certified-pairs.tsv) — meaning verified by
     a full round trip; the gold layer
  2. woven transfer edges   (chain-web.tsv) — seed -> endpoint chains
  3. all-step connections   (chain-web-full.tsv) — every interior hop of
     every chain, so partial/indirect transfers are reachable too
Each carries its full chain provenance (key = touche ~ douche = shower ...).
Then the LLM fluency layer (DeepSeek via llm_layer, if DEEPSEEK_API_KEY is
set) arranges the chain-chosen fragments into a grammatical target line —
the engine owns the sound+meaning chains, the LLM owns only word order.

    python chain_engine.py --text "the sea is cold and the moon is bright"
    python chain_engine.py --llm --show-work --text "..."

This is en-fr (the fully woven pair). Other pairs: run multilang.py then
weave.py for that pair to produce its own chain-web, then point --webdir at it.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

EN_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "or",
           "in", "on", "at", "to", "it", "its", "be", "by", "with", "as",
           "that", "this", "but", "for", "from", "i", "you", "we", "they"}


def load_web(webdir="."):
    """Returns src_word -> ranked list of {tgt, quality, chain, tier}."""
    transfers = defaultdict(list)

    # 1. loop-certified (gold): src may be either side of the certified pair
    p = os.path.join(webdir, "loop-certified-pairs.tsv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                transfers[r["en"]].append(
                    {"tgt": r["fr"], "quality": 1.0 + int(r["certifications"]) / 10,
                     "chain": r["example_loop"], "tier": "loop-certified"})

    # 2. woven transfer edges
    p = os.path.join(webdir, "chain-web.tsv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                transfers[r["src"]].append(
                    {"tgt": r["dst"], "quality": float(r["quality"]),
                     "chain": r["chain"], "tier": f"chain/{r['hops']}hop"})

    # 3. all-step interior connections (en:* -> fr:* only)
    p = os.path.join(webdir, "chain-web-full.tsv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                a, b = r["a"], r["b"]
                if a.startswith("en:") and b.startswith("fr:"):
                    transfers[a[3:]].append(
                        {"tgt": b[3:], "quality": float(r["quality"]) * 0.95,
                         "chain": r["subchain"], "tier": "all-step"})
    for w in transfers:
        transfers[w].sort(key=lambda x: -x["quality"])
    return transfers


def main():
    ap = argparse.ArgumentParser(description="Chain-web-routed homophonic engine")
    ap.add_argument("--text", default=None)
    ap.add_argument("--webdir", default=".")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="live alternation-chain search for words missing from "
                         "the precomputed web (fullest form; builds graph ~8min)")
    ap.add_argument("--show-work", action="store_true")
    args = ap.parse_args()

    web = load_web(args.webdir)
    live = None
    if args.live:
        import chain_translate, chain_game
        print("building live chain graph (one-time)...", file=sys.stderr)
        _edges, _ = chain_game.build_graph()
        live = (chain_translate, _edges)
    n_gold = sum(1 for w in web for t in web[w] if t["tier"] == "loop-certified")
    print(f"chain-web loaded: {len(web)} source words, {n_gold} gold transfers",
          file=sys.stderr)

    text = args.text or sys.stdin.read()
    lines = [ln for ln in text.splitlines() if ln.strip()] or [text]

    arranger = None
    if args.llm:
        import llm_layer
        if llm_layer.available():
            arranger = llm_layer
        else:
            print("  [--llm: no DEEPSEEK_API_KEY in env; raw chain output only]")

    out = []
    for ln in lines:
        words = [w.lower().strip(".,!?;:\"'") for w in ln.split()]
        rendered, detail, llm_opts = [], [], []
        for w in words:
            if not w:
                continue
            if w in EN_STOP:
                rendered.append("·")
                continue
            options = web.get(w, [])
            if not options and live:
                ct, edges = live
                eps = ct.fr_endpoints(edges, w)
                ranked = sorted(((q, fw, p) for fw, (rank, q, ln, p) in eps.items()
                                 if q >= 0.78), reverse=True)[:5]
                options = [{"tgt": fw, "quality": q, "chain": p, "tier": "live-chain"}
                           for q, fw, p in ranked]
            if not options:
                rendered.append(f"[{w}]")
                detail.append(f"      {w}: no chain transfer")
                continue
            best = options[0]
            rendered.append(best["tgt"])
            detail.append(f"      {w} → {best['tgt']}  [{best['tier']}, q{best['quality']:.2f}]")
            detail.append(f"          chain: {best['chain']}")
            llm_opts.append({"src_word": w,
                             "renderings": [{"tgt": o["tgt"], "sound": min(1.0, o["quality"])}
                                            for o in options[:5]]})
        line = " ".join(t for t in rendered if t != "·")
        out.append(f"ORIGINAL    {ln.strip()}")
        out.append(f"CHAIN-WEB   {line[:1].upper() + line[1:]}")
        if arranger and llm_opts:
            res = arranger.arrange_line(ln.strip(), "French", llm_opts)
            if res and res.get("line"):
                out.append(f"FLUENT      {res['line']}")
                if res.get("gloss"):
                    out.append(f"  (gloss: {res['gloss']})")
        if args.show_work:
            out.extend(detail)
        out.append("")
    text_out = "\n".join(out)
    print(text_out)
    with open("chain-engine-demo.txt", "w") as f:
        f.write(text_out + "\n")


if __name__ == "__main__":
    main()
