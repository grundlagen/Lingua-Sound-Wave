"""Tiered homophonic+semantic translation engine.

Input: an English phrase. For each content word, render it in French that SOUNDS
like the English and MEANS the same, preferring sound fidelity and degrading
gracefully:

  TIER S  sound-fidelity 1-hop: the word has a French homophone that already
          preserves meaning (loop-certified S-tier, combo>=0.90). Zero inflation,
          highest fidelity -- the gold case.
  TIER A  homophone + synonym chain: shortest homophonic route (A-tier sound
          >=0.80 plus synonym hops, >=1 sound hop) to a French word meaning the
          source (its translation OR a synonym of it). Inflation = extra words.
  TIER P  poetic periphrasis: no homophonic route to the exact meaning, so use a
          metaphor -- the semantically nearest French expression (prefer
          multiword) that IS homophonically reachable. "small lizard / frog" idea.
  literal fallback: plain translation, tagged (no homophony) -- last resort.

Uses the full-scope v7 graph + the same multilingual embeddings the weave used
(node-vecs.npy) for semantic matching and the poetic tier.

Run: python translate_engine.py "the key is gold"
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import deque

import numpy as np

SOUND, MEANING = "S", "M"


def valid_nodes(ids, french_vocab):
    """Keep only real EN / FR words. English side: zipf gate (sound bridges).
    French side: must be an ACTUAL French translation word (MUSE/dict French
    vocab) + zipf gate -- this is the hard language gate that drops the
    multilingual junk the multilingual embedding pulled in (ključ, amigos,
    chaves, winstead, makina, ...) which mere corpus-frequency lets through."""
    from wordfreq import zipf_frequency
    ok = set()
    for n in ids:
        w = n[3:]
        toks = w.split()
        if n.startswith("en:"):
            if all(zipf_frequency(t, "en") >= 1.5 for t in toks):
                ok.add(n)
        else:  # fr
            if not all(zipf_frequency(t, "fr") >= 2.5 for t in toks):
                continue
            if len(toks) == 1 and w not in french_vocab:
                continue          # single French word must be a real translation
            ok.add(n)
    return ok


def load():
    edges = pickle.load(open("graph-v7u.pkl", "rb"))["edges"]
    lcS = {}
    for i, line in enumerate(open("loop-certified-pairs-v7u-S.tsv", encoding="utf-8")):
        if i == 0:
            continue
        en, fr, *_ = line.split("\t")
        lcS.setdefault(en, fr)
    vecs = np.load("node-vecs.npy")
    ids = json.load(open("node-ids.json"))
    idx = {n: i for i, n in enumerate(ids)}
    en_mask = np.array([n.startswith("en:") for n in ids])
    fr_mask = np.array([n.startswith("fr:") for n in ids])
    tgt = {}
    french_vocab = set()
    with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                tgt.setdefault(p[0], set()).add(p[1])
                french_vocab.add(p[1])
    valid = valid_nodes(ids, french_vocab)
    return edges, lcS, vecs, ids, idx, en_mask, fr_mask, tgt, valid


def homophonic_adj(edges, valid):
    """sound+synonym edges, restricted to real EN/FR word nodes."""
    adj = {}
    for n, lst in edges.items():
        if n not in valid:
            continue
        keep = [(m, lab, fam) for m, q, lab, fam in lst
                if lab in ("≈", "~") and m in valid]
        if keep:
            adj[n] = keep
    return adj


def route(adj, src, targets, cap=8):
    """shortest path src -> any target, >=1 sound hop, no U-turns; [(node,lab)]."""
    if src not in adj:
        return None
    start = (src, False)
    prev = {start: (None, None)}
    q = deque([(src, False, 0, None)])      # node, sound_used, depth, parent
    while q:
        node, su, d, par = q.popleft()
        if d >= cap:
            continue
        for m, lab, fam in adj.get(node, ()):
            if m == par:                    # no immediate U-turn
                continue
            nsu = su or (fam == SOUND)
            st = (m, nsu)
            if st not in prev:
                prev[st] = ((node, su), lab)
                if m in targets and nsu:
                    path = [(m, lab)]
                    cur = (node, su)
                    while cur is not None:
                        pcur, plab = prev[cur]
                        path.append((cur[0], plab))
                        cur = pcur
                    return path[::-1]
                q.append((m, nsu, d + 1, node))
    return None


def nearest_en(word, model, vecs, ids, idx, en_valid_mask):
    """resolve an input word to the nearest real English node."""
    n = f"en:{word}"
    if n in idx and en_valid_mask[idx[n]]:
        return n
    v = model.encode([word], normalize_embeddings=True)[0].astype(np.float32)
    sims = vecs @ v
    sims[~en_valid_mask] = -1
    return ids[int(sims.argmax())]


def translate_word(w, ctx):
    (edges, lcS, vecs, ids, idx, en_mask, fr_mask, tgt, adj, model,
     en_valid_mask, fr_valid_mask) = ctx
    en_node = nearest_en(w, model, vecs, ids, idx, en_valid_mask)
    base = en_node[3:]

    # TIER S -------------------------------------------------------------
    if base in lcS:
        return ("S", lcS[base], [(f"en:{base}", None), (f"fr:{lcS[base]}", "≈")], 0)

    # meaning-equivalent target set: translations + their synonym closure
    targets = {f"fr:{t}" for t in tgt.get(base, ())} & set(adj)
    syn_targets = set(targets)
    for t in list(targets):
        syn_targets |= {m for m, lab, fam in adj.get(t, ()) if lab == "~" and m.startswith("fr:")}

    # TIER A -- shortest homophonic path to the meaning (long is OK) ------
    p = route(adj, en_node, syn_targets, cap=14)
    if p:
        return ("A", p[-1][0][3:], p, len(p) - 2)

    # TIER P (poetic periphrasis) ---------------------------------------
    v = vecs[idx[en_node]]
    sims = vecs @ v
    sims[~fr_valid_mask] = -1
    near = np.argpartition(-sims, 400)[:400]
    metaphor = {ids[int(j)] for j in near if sims[j] >= 0.42} & set(adj)
    pp = route(adj, en_node, metaphor, cap=12)
    if pp:
        return ("P", pp[-1][0][3:], pp, len(pp) - 2)

    # literal fallback ---------------------------------------------------
    lit = next(iter(tgt.get(base, {"?"})))
    return ("lit", lit, [(f"en:{base}", None), (f"fr:{lit}", "=")], 0)


def render_chain(path):
    """labeled path -> readable 'a ≈ b ~ c = d' with bare words."""
    out = [path[0][0].split(":", 1)[1]]
    for node, lab in path[1:]:
        out.append(f"{lab} {node.split(':', 1)[1]}")
    return " ".join(out)


def main():
    sent = sys.argv[1] if len(sys.argv) > 1 else "the key is gold"
    edges, lcS, vecs, ids, idx, en_mask, fr_mask, tgt, valid = load()
    adj = homophonic_adj(edges, valid)
    valid_mask = np.array([n in valid for n in ids])
    en_valid_mask = en_mask & valid_mask
    fr_valid_mask = fr_mask & valid_mask
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    ctx = (edges, lcS, vecs, ids, idx, en_mask, fr_mask, tgt, adj, model,
           en_valid_mask, fr_valid_mask)

    words = [w.strip(".,!?;:").lower() for w in sent.split() if w.strip(".,!?;:")]
    print(f'INPUT  : "{sent}"\n')
    print("=== shortest homophonic path per word (every hop written out) ===")
    endpoints, stream, total_hops = [], [], 0
    for w in words:
        tier, fr, path, infl = translate_word(w, ctx)
        endpoints.append(fr)
        total_hops += len(path) - 1
        # the written homophonic stream = the spoken fragments along the path
        stream.extend(node.split(":", 1)[1] for node, _ in path[1:])
        print(f"\n  [{tier}] {w}  ({len(path)-1} hops):")
        print(f"      {render_chain(path)}")
    print("\n=== WRITTEN-OUT homophonic rendering (all intermediate hops) ===")
    print("  " + " ".join(stream))
    print("\n=== final French (meaning-equivalent endpoints) ===")
    print(f'  "{" ".join(endpoints)}"')
    print(f"\n{len(words)} source words -> {total_hops} total hops "
          f"({len(stream)} written fragments). tiers: S 1-hop, A homophone+synonym "
          f"chain, P poetic periphrasis, lit literal.")


if __name__ == "__main__":
    main()
