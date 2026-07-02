"""EVAL HARNESS -- the data we actually want, captured every training round.

Runs a FROZEN evaluation against any checkpoint (or base model) and appends
one row per round to selflearn/RESULTS.tsv plus verified samples to
selflearn/SAMPLES.md. Frozen = never trained on, never changed, so the curve
across rounds is the real signal.

What it measures (all judge-scored, never self-reported):
  word tier-rate    50 frozen EN words -> model FR; % with combo>=0.60 and
                    >=0.75 (the DUAL-A/S bands)
  line joint-rate   10 frozen corpus lines -> model FR; % with sound>=0.55
                    AND meaning>=0.45 (Rooten band), + mean sound/meaning
  degeneration      % of outputs that fail the Lexique gate (franglais/junk)

Run: python eval_harness.py --model <ckpt_dir_or_hf_name> --round N
     (called automatically by train_selflearn.py after each round)
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))          # matcher, semantic_cosine

import matcher
from semantic_cosine import semantic_cosine

RESULTS = os.path.join(_HERE, "RESULTS.tsv")
SAMPLES = os.path.join(_HERE, "SAMPLES.md")

FROZEN_WORDS = ["thunder", "freedom", "window", "mountain", "shadow", "river",
                "garden", "winter", "morning", "silver", "hunger", "wisdom",
                "father", "mother", "brother", "water", "fire", "stone",
                "bread", "night", "dream", "heart", "voice", "storm", "field",
                "horse", "bird", "snow", "rain", "wind", "star", "moon",
                "light", "dark", "gold", "iron", "wood", "glass", "salt",
                "honey", "wolf", "bear", "fish", "tree", "leaf", "root",
                "seed", "corn", "wine", "milk"]
FROZEN_LINES = ["the sea remembers every ship",
                "we call to the moon and she answers",
                "my sorrow sleeps in a deep well",
                "bless the dawn that made us free",
                "the wolf waits at the garden gate",
                "cold rain falls on the old stone bridge",
                "my brother sings to the winter stars",
                "the river carries our names to the sea",
                "less debt, less mess, more soup",
                "a bird of gold sleeps in the snow"]
PROMPT = ("Rewrite this English as French that sounds the same when read "
          "aloud and stays coherent French: {en}")


def combo(en, fr):
    try:
        qi, ci = matcher.g2p(en, "en"), matcher.g2p(fr, "fr")
        return 0.5 * matcher._ngram_channel(qi, ci) + 0.5 * matcher._feat_channel(qi, ci)
    except Exception:
        return 0.0


_FRV = None
def frgate(text):
    global _FRV
    if _FRV is None:
        _FRV = set()
        lex = os.path.join(os.path.dirname(_HERE), "data", "lexique.tsv")
        for line in open(lex, encoding="utf-8", errors="ignore"):
            w = line.split("\t")[0].strip().lower()
            if w:
                _FRV.add(w)
    ws = [w.strip(",.;:!?'").lower() for w in text.replace("'", "' ").split()]
    return all((not w) or w in _FRV for w in ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--k", type=int, default=4, help="samples per prompt, best kept")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto",
                                                     device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                     device_map="auto")
    model.eval()

    def gen(prompt, k):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, do_sample=True, temperature=0.9, top_p=0.95,
                                 num_return_sequences=k, max_new_tokens=40,
                                 pad_token_id=tok.pad_token_id)
        return [tok.decode(o[ids.shape[1]:], skip_special_tokens=True)
                .strip().split("\n")[0] for o in out]

    # ---- word tier-rate
    w_best, junk = [], 0
    for en in FROZEN_WORDS:
        cands = [c for c in gen(PROMPT.format(en=en), args.k) if c]
        scored = []
        for c in cands:
            if not frgate(c):
                junk += 1
                continue
            scored.append((combo(en, c), c))
        w_best.append(max(scored)[0] if scored else 0.0)
    n_gen = len(FROZEN_WORDS) * args.k
    dualA = sum(s >= 0.60 for s in w_best) / len(w_best)
    dualS = sum(s >= 0.75 for s in w_best) / len(w_best)

    # ---- line joint-rate
    l_s, l_m, band, samples = [], [], 0, []
    for en in FROZEN_LINES:
        cands = [c for c in gen(PROMPT.format(en=en), args.k) if c and frgate(c)]
        best = None
        for c in cands:
            s = combo(en, c)
            m = max(0.0, semantic_cosine(en, c))
            j = (s * m) ** 0.5
            if best is None or j > best[0]:
                best = (j, s, m, c)
        if best:
            _, s, m, c = best
            l_s.append(s); l_m.append(m)
            band += (s >= 0.55 and m >= 0.45)
            samples.append((en, c, s, m))
        else:
            l_s.append(0.0); l_m.append(0.0)

    ms = sum(l_s) / len(l_s)
    mm = sum(l_m) / len(l_m)
    br = band / len(FROZEN_LINES)
    jr = junk / max(1, n_gen)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", encoding="utf-8") as f:
        if new:
            f.write("time\tround\tmodel\tword_dualA\tword_dualS\tline_band\t"
                    "line_sound\tline_meaning\tjunk_rate\n")
        f.write(f"{now}\t{args.round}\t{os.path.basename(args.model.rstrip('/'))}\t"
                f"{dualA:.3f}\t{dualS:.3f}\t{br:.3f}\t{ms:.3f}\t{mm:.3f}\t{jr:.3f}\n")
    with open(SAMPLES, "a", encoding="utf-8") as f:
        f.write(f"\n## round {args.round} @ {now}\n")
        for en, c, s, m in samples[:6]:
            f.write(f"- `{en}` → **{c}**  (snd {s:.2f} / mng {m:.2f})\n")
    print(f"EVAL round {args.round}: word A/S {dualA:.0%}/{dualS:.0%}  "
          f"line band {br:.0%} (snd {ms:.2f} mng {mm:.2f})  junk {jr:.0%}")


if __name__ == "__main__":
    main()
