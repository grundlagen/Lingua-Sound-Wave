#!/usr/bin/env python3
"""
THREE-AGENT HOMOPHONE REFINERY — Iterative competition with synonym drift.

Agent A (Forward):   English → French homophone text
Agent B (Reverse):   French text → semantically equivalent English (NOT homophone-reversing)
Agent C (Judge):     Compares original English vs Agent B's English. Scores meaning preservation.

THE LOOP (4-5 iterations):
  1. Agent A converts English to French homophones
  2. Agent B reads the French and produces NEW English that preserves meaning
  3. Agent C judges: how close is Agent B's English to the original?
  4. If Agent B's English differs, it becomes the NEW input for Agent A
  5. This naturally drifts toward words with better homophone matches:
     "ocean" might become "sea" because "sea"→"scient" is in the DB but "ocean"→? is not
     "recalls" → "remembers" because "remembers"→"mêmes" exists

SYNONYM ENGINE: Uses the muse-pivot-syn.tsv (51k synonym edges) to suggest
alternative English words that might have better homophone matches.

Run: python three_agent_refinery.py
"""

import json, os, random
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════
print("Loading resources...")

# ── Homophone DB (6,143 EN→FR pairs) ──
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]

homophone_db = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in homophone_db or q > homophone_db[en][1]:
            homophone_db[en] = (fr, q)

# Reverse: FR → [(EN, quality), ...]
fr_to_ens = defaultdict(list)
for en, (fr, q) in homophone_db.items():
    fr_to_ens[fr].append((en, q))

# ── English synonyms (muse-pivot-syn, 51k edges) ──
syn_en = defaultdict(set)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:])
        syn_en[b[3:]].add(a[3:])

print(f"  Homophone DB: {len(homophone_db)} pairs")
print(f"  Synonyms: {sum(len(v) for v in syn_en.values())} edges")

# ── Similarity ──
def word_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))

_SEM = None
def semantic_sim(a, b):
    global _SEM
    if _SEM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except: _SEM = False
    if _SEM is False: return word_sim(a, b) * 0.5 + 0.5
    try:
        v = _SEM.encode([a, b], normalize_embeddings=True)
        return float(v[0] @ v[1])
    except: return 0.5

# ═══════════════════════════════════════════════════════════════
# AGENT A: English → French homophone
# ═══════════════════════════════════════════════════════════════
def agent_A(en_text):
    """Convert English to French homophones. Returns list of (en_w, fr_w, quality)."""
    words = [w.lower().strip(".,;:!?'\"") for w in en_text.split()
             if w.strip(".,;:!?'\"")]
    result = []
    for w in words:
        if w in homophone_db:
            fr, q = homophone_db[w]
            result.append((w, fr, q, "direct"))
        else:
            # Nearest match
            best = max(homophone_db.keys(), key=lambda k: word_sim(w, k))
            fr, q = homophone_db[best]
            result.append((w, fr, q * 0.85, f"near:{best}"))
    return result

# ═══════════════════════════════════════════════════════════════
# AGENT B: French → English (meaning-preserving, NOT homophone-reversing)
# ═══════════════════════════════════════════════════════════════
def agent_B(fr_words, original_en_words, homophone_pairs):
    """
    Agent B (Reverse): Given French homophone text + the original pairs,
    recover English that preserves meaning.
    
    Strategy: For each FR word, check what English words it COULD mean.
    If the original EN word is in the candidate set → return it.
    If not → find the candidate with best synonym-path to the original.
    """
    result = []
    for i, fr_w in enumerate(fr_words):
        orig_en = original_en_words[i] if i < len(original_en_words) else ""
        
        # Look up all EN words that produced this FR word (from current iteration's pairs)
        possible = fr_to_ens.get(fr_w, [])
        
        if not possible:
            result.append(orig_en)  # keep original — no information
            continue
        
        # Does the original appear in the candidate set?
        candidates = [en for en, _ in possible]
        if orig_en in candidates:
            result.append(orig_en)  # direct match — confident
            continue
        
        # Original not in candidates. Find best semantic match.
        best_sim, best_en = 0, candidates[0]
        for en_c in candidates:
            if en_c in syn_en.get(orig_en, set()):
                best_en, best_sim = en_c, 0.9
                break
            sim = semantic_sim(en_c, orig_en)
            if sim > best_sim:
                best_sim, best_en = sim, en_c
        
        result.append(best_en)
    
    return " ".join(result)

# ═══════════════════════════════════════════════════════════════
# AGENT C: Judge — compare original vs recovered English
# ═══════════════════════════════════════════════════════════════
def agent_C(original_en, recovered_en):
    """
    Judge how well Agent B preserved the original meaning.
    Returns: score 0-1, and a list of words that changed.
    """
    orig_words = original_en.lower().split()
    rec_words = recovered_en.lower().split()
    
    # Semantic similarity
    sem = semantic_sim(original_en, recovered_en)
    
    # Exact word match ratio
    n = min(len(orig_words), len(rec_words))
    exact = sum(1 for i in range(n) if orig_words[i] == rec_words[i])
    exact_ratio = exact / max(1, n)
    
    # Words that changed
    changed = []
    for i in range(n):
        if orig_words[i] != rec_words[i]:
            changed.append((orig_words[i], rec_words[i]))
    
    score = 0.5 * sem + 0.5 * exact_ratio
    
    return {
        "score": round(score, 3),
        "semantic": round(sem, 3),
        "exact": round(exact_ratio, 3),
        "changed": changed,
        "recovered": recovered_en,
    }

# ═══════════════════════════════════════════════════════════════
# SYNONYM ENGINE — suggest better English words
# ═══════════════════════════════════════════════════════════════
def synonym_suggest(word):
    """Find synonyms that have BETTER homophone matches in the DB."""
    if word not in syn_en:
        return []
    
    candidates = []
    for syn in syn_en[word]:
        if syn in homophone_db:
            fr, q = homophone_db[syn]
            candidates.append((syn, q, fr))
    
    candidates.sort(key=lambda x: -x[1])
    return candidates[:5]

# ═══════════════════════════════════════════════════════════════
# REFINERY LOOP — 5 iterations of competition
# ═══════════════════════════════════════════════════════════════
def refine(initial_english, max_iterations=5):
    """
    Three-agent iterative refinement.
    English input drifts toward words with better homophone matches
    while preserving semantic meaning.
    """
    current_en = initial_english
    history = []
    
    for iteration in range(1, max_iterations + 1):
        # ── Agent A: English → French homophones ──
        a_result = agent_A(current_en)
        fr_words = [fr for _, fr, _, _ in a_result]
        fr_text = " ".join(fr_words)
        
        # ── Agent B: French → Recovered English ──
        orig_words = initial_english.lower().split()
        recovered_en = agent_B(fr_words, orig_words, a_result)
        
        # ── Agent C: Judge ──
        judgment = agent_C(initial_english, recovered_en)
        
        # ── Synonym drift: for words with weak homophones, try synonyms ──
        drift_applied = False
        en_words = current_en.lower().split()
        new_en_words = list(en_words)
        
        for i, (en_w, fr_w, q, src) in enumerate(a_result):
            if q < 0.90 or src.startswith("near"):
                syns = synonym_suggest(en_w)
                if syns and syns[0][1] > q:
                    new_en_words[i] = syns[0][0]
                    drift_applied = True
        
        # ── Use Agent B's recovery as next input if significant drift ──
        if judgment["score"] < 0.70 and judgment["changed"]:
            new_en_words = recovered_en.lower().split()
            drift_applied = True
        
        next_en = " ".join(new_en_words)
        
        history.append({
            "iteration": iteration,
            "input_en": current_en,
            "french": fr_text,
            "recovered_en": recovered_en,
            "judgment": judgment,
            "drifted_to": next_en if drift_applied else None,
        })
        
        # Show iteration
        changed_str = ", ".join(f"{a}→{b}" for a,b in judgment["changed"][:5])
        print(f"  [{iteration}] EN: {current_en[:60]}...")
        print(f"       FR: {fr_text[:60]}...")
        print(f"       RE: {recovered_en[:60]}...")
        print(f"       Judge: score={judgment['score']:.3f} "
              f"sem={judgment['semantic']:.3f} exact={judgment['exact']:.3f}")
        if changed_str:
            print(f"       Changed: {changed_str}")
        if drift_applied:
            print(f"       DRIFT → {next_en[:60]}...")
        print()
        
        if drift_applied:
            # Check for cycles: if next_en is same as a previous iteration, stop
            if any(h["input_en"] == next_en for h in history):
                print(f"       Cycle detected — stopping.")
                break
            current_en = next_en
        else:
            print(f"       Converged — no further improvement.")
            break
    
    return history

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        "the ocean remembers every vessel that ever sailed",
        "a small stream becomes a mighty river",
        "she wandered through the forest at twilight",
    ]
    
    for test in tests:
        print(f"\n{'='*70}")
        print(f"THREE-AGENT REFINERY: \"{test}\"")
        print(f"{'='*70}")
        
        history = refine(test, max_iterations=5)
        
        # Final summary
        if history:
            last = history[-1]
            print(f"  FINAL after {len(history)} iterations:")
            print(f"  Input:     {test}")
            print(f"  Drifted:   {last['input_en']}")
            print(f"  French:    {last['french']}")
            print(f"  Recovered: {last['recovered_en']}")
            print(f"  Score:     {last['judgment']['score']:.3f}")
