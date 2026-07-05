"""chain_paragraph.py — translate a paragraph through chain hops.

Uses the FULL chain-hop system: every content word transfers to French
through alternation chains (interleaved sound ≈ and meaning =/~ hops).
The output includes all intermediary hops so you can see the chain
inflation. Where no chain works, falls back to the phonetic decoder
(trie beam search).

This is a standalone script that reads dictionary-v5.json, MUSE, and
the Lexique trie. Does not modify existing code.

    python chain_paragraph.py "the pale moon lights the quiet sea"
    python chain_paragraph.py --show-chains "..."
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict

import numpy as np
from wordfreq import zipf_frequency

import chain_game
import chain_translate as ct
import matcher
import phonetic_decoder as pd
from lexicon_g2p import clean_ipa

EN_STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "and", "or",
           "in", "on", "at", "to", "it", "its", "be", "by", "with", "as",
           "that", "this", "but", "for", "from", "i", "you", "he", "she",
           "we", "they", "me", "my", "his", "her", "our", "your"}


def espeak_ipa(text: str) -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", "en-us", text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())


def decoder_fallback(word: str, root, model, anchor_vec, top_n=12):
    """phonetic_decoder beam search as fallback when chains fail."""
    ipa = espeak_ipa(word)
    word_len = len(matcher._segs(matcher._canonical(ipa)))
    min_cov = 0.55 if word_len >= 5 else 0.70
    max_exp = 1 if word_len >= 5 else 0
    candidates = [c for c in pd.decode(ipa, root, top_n=top_n, max_words=6)
                  if c["coverage"] >= min_cov
                  and c["expensive_deletions"] <= max_exp]
    if not candidates:
        return None
    fr_texts = [c["fr"] for c in candidates]
    vecs = np.asarray(model.encode(fr_texts, normalize_embeddings=True,
                                   show_progress_bar=False))
    best = None
    for c, v in zip(candidates, vecs):
        sem = max(0.0, float(np.dot(anchor_vec, v)))
        score = c["similarity"] * (0.3 + 0.7 * sem)
        if best is None or score > best["score"]:
            best = {"tgt": c["fr"], "sound": c["similarity"], "sem": sem,
                    "score": score, "method": "decoder",
                    "chain": f"[decoder] en:{word} →ipa→ {c['fr']}"}
    return best


_MUSE_EN2FR: dict[str, list[str]] | None = None

def _load_muse():
    global _MUSE_EN2FR
    if _MUSE_EN2FR is not None:
        return
    _MUSE_EN2FR = defaultdict(list)
    try:
        with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    _MUSE_EN2FR[p[0]].append(p[1])
    except FileNotFoundError:
        pass

def _synonyms_local(word: str) -> list[str]:
    """Offline synonyms: EN->FR->EN pivot through MUSE."""
    _load_muse()
    frs = set(_MUSE_EN2FR.get(word, []))
    if not frs:
        return []
    fr2en = defaultdict(list)
    for en, fr_list in _MUSE_EN2FR.items():
        for fr in fr_list:
            if fr in frs and en != word:
                fr2en[en] = True
    return sorted(fr2en.keys())[:6]


def transfer_word(word: str, edges, model, anchor_bar=0.30):
    """Transfer one word via chain hops, with synonym expansion."""
    wv = model.encode([word], normalize_embeddings=True,
                      show_progress_bar=False)[0]

    seeds = [word] + _synonyms_local(word)
    all_candidates = []
    for seed in seeds:
        eps = ct.fr_endpoints(edges, seed)
        for fw, (rank, q, hops, chain) in eps.items():
            if q >= 0.70:
                all_candidates.append((fw, q, hops, chain, seed))

    if not all_candidates:
        return None

    fr_words = list({c[0] for c in all_candidates})
    fv = dict(zip(fr_words,
                  model.encode(fr_words, normalize_embeddings=True,
                               show_progress_bar=False)))
    best = None
    for fw, q, hops, chain, seed in all_candidates:
        anchor = max(0.0, float(np.dot(wv, fv[fw])))
        score = q * (0.25 + 0.75 * anchor)
        via_note = "" if seed == word else f" (via synonym '{seed}')"
        if anchor >= anchor_bar * 0.3 and (best is None or score > best["score"]):
            best = {"tgt": fw, "sound": q, "sem": anchor, "score": score,
                    "hops": hops, "chain": chain + via_note, "method": "chain"}
    return best


def translate_paragraph(text: str, edges, model, root, show_chains=False):
    """Translate a full paragraph, word by word, through the chain system."""
    words = [w.lower().strip(".,!?;:\"'()") for w in text.split()]
    results = []

    for w in words:
        if not w:
            continue
        if w in EN_STOP:
            results.append({"src": w, "tgt": w, "method": "stop",
                            "chain": "", "sound": 0, "sem": 0, "score": 0})
            continue

        # Try chain transfer first
        b = transfer_word(w, edges, model)
        if b:
            results.append({"src": w, **b})
            continue

        # Fall back to decoder
        anchor = model.encode([w], normalize_embeddings=True,
                              show_progress_bar=False)[0]
        b = decoder_fallback(w, root, model, anchor)
        if b:
            results.append({"src": w, **b})
            continue

        results.append({"src": w, "tgt": f"[{w}]", "method": "miss",
                        "chain": "", "sound": 0, "sem": 0, "score": 0})

    return results


def format_output(text: str, results: list[dict], show_chains: bool) -> str:
    lines = []
    lines.append(f"ORIGINAL    {text}")
    lines.append("")

    # Build the FR line
    fr_parts = []
    for r in results:
        if r["method"] == "stop":
            continue
        fr_parts.append(r["tgt"])
    fr_line = " ".join(fr_parts)
    lines.append(f"HOMOPHONIC  {fr_line}")
    lines.append("")

    # Stats
    chain_count = sum(1 for r in results if r["method"] == "chain")
    decoder_count = sum(1 for r in results if r["method"] == "decoder")
    miss_count = sum(1 for r in results if r["method"] == "miss")
    content = [r for r in results if r["method"] not in ("stop",)]
    avg_sound = sum(r["sound"] for r in content) / max(1, len(content))
    avg_sem = sum(r["sem"] for r in content) / max(1, len(content))

    lines.append(f"STATS       {chain_count} chain-transferred, "
                 f"{decoder_count} decoder-fallback, {miss_count} missed")
    lines.append(f"            avg sound={avg_sound:.3f}  avg meaning={avg_sem:.3f}")
    lines.append("")

    # Per-word detail
    lines.append("WORD-BY-WORD:")
    for r in results:
        if r["method"] == "stop":
            continue
        method_tag = r["method"].upper()
        if r["method"] == "chain":
            lines.append(f"  {r['src']:20s} → {r['tgt']:20s}  "
                         f"[{method_tag} {r.get('hops',0)}h]  "
                         f"sound={r['sound']:.3f}  meaning={r['sem']:.3f}")
        else:
            lines.append(f"  {r['src']:20s} → {r['tgt']:20s}  "
                         f"[{method_tag}]  "
                         f"sound={r['sound']:.3f}  meaning={r['sem']:.3f}")
        if show_chains and r["chain"]:
            lines.append(f"    chain: {r['chain']}")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Chain-hop paragraph translation")
    ap.add_argument("text", nargs="?",
                    default="the pale moon lights the quiet sea")
    ap.add_argument("--show-chains", "-c", action="store_true",
                    help="show full chain paths for each word")
    args = ap.parse_args()

    print("building 3-layer graph (sound ≈, translation =, semantic ~)...",
          file=sys.stderr)
    edges, sem_neigh = chain_game.build_graph()
    edge_count = sum(len(v) for v in edges.values()) // 2
    print(f"  graph: {len(edges)} nodes, {edge_count} edges", file=sys.stderr)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    print("building FR pronunciation trie...", file=sys.stderr)
    root = pd.build_trie(min_zipf=2.0)

    print("translating...\n", file=sys.stderr)
    results = translate_paragraph(args.text, edges, model, root,
                                  show_chains=args.show_chains)
    output = format_output(args.text, results, args.show_chains)
    print(output)

    with open("chain-paragraph-demo.txt", "w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nwritten to chain-paragraph-demo.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
