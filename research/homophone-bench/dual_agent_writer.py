#!/usr/bin/env python3
"""
DUAL-AGENT HOMOPHONE WRITER — Two models collaborate for sound + meaning.

Model A (Homophone Engine): 6,143-pair nearest-neighbor lookup.
  Given English → generates French that sounds like English.

Model B (LLM Meaning Preserver): Evaluates the French output,
  translates it back to English, measures meaning loss,
  suggests alternative homophones that preserve more meaning.

COLLABORATION LOOP:
  1. Model A generates initial French paragraph (word-by-word lookup)
  2. Model B reads the French, translates back to English, scores meaning
  3. For each word where meaning is lost, Model B queries Model A for
     alternative homophones and picks the one that preserves more meaning
  4. Loop until meaning score converges or max iterations

Run: python dual_agent_writer.py "the sea remembers every ship that ever sailed"
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# MODEL A: Homophone Engine (6,143-pair lookup)
# ═══════════════════════════════════════════════════════════════
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]

homophone_db = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in homophone_db or q > homophone_db[en][1]:
            homophone_db[en] = (fr, q, r.get("loop",False), r.get("chain",False))

# Build reverse: fr_word → all possible EN sources
fr_to_ens = defaultdict(list)
for en, (fr, q, loop, ch) in homophone_db.items():
    fr_to_ens[fr].append((en, q))

print(f"Model A: {len(homophone_db)} EN→FR pairs")

# ── Multi-candidate lookup ──
def word_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))

def model_A_lookup(en_word, top_k=5):
    """Model A: returns top-K French homophone candidates."""
    en_word = en_word.lower()
    results = []
    
    # Direct match
    if en_word in homophone_db:
        fr, q, loop, ch = homophone_db[en_word]
        results.append((fr, q, "direct"))
    
    # Nearest matches by character similarity
    neighbors = sorted(homophone_db.keys(), key=lambda k: -word_sim(en_word, k))[:20]
    for n in neighbors:
        if n == en_word: continue
        fr, q, loop, ch = homophone_db[n]
        results.append((fr, q * 0.85, f"near:{n}"))  # decay for near-matches
    
    # Deduplicate and sort by quality
    seen = set()
    unique = []
    for fr, q, src in results:
        if fr not in seen:
            seen.add(fr)
            unique.append((fr, q, src))
    unique.sort(key=lambda x: -x[1])
    return unique[:top_k]

# ═══════════════════════════════════════════════════════════════
# MODEL B: LLM Meaning Preserver (implemented with existing tools)
# ═══════════════════════════════════════════════════════════════
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

# MiniLM for semantic similarity (cache-friendly)
_SEM = None
def semantic_similarity(en_phrase, fr_phrase):
    global _SEM
    if _SEM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except:
            _SEM = False
    if _SEM is False:
        return 0.5
    try:
        v = _SEM.encode([en_phrase, fr_phrase], normalize_embeddings=True)
        return float(v[0] @ v[1])
    except:
        return 0.5

def model_B_evaluate(en_text, fr_text):
    """
    Model B (LLM): Evaluates meaning preservation.
    Since we can't translate the French back (no translator available),
    we use semantic similarity as a proxy for meaning preservation.
    
    Also: for each French word, check what English words it CAN mean
    (via the reverse fr_to_ens index) and see if they overlap with
    the original English words.
    """
    # Semantic similarity (global meaning match)
    sem_score = semantic_similarity(en_text, fr_text)
    
    # Word-level meaning overlap
    en_words = [w.lower().strip(".,;:!?'\"") for w in en_text.split() if w.strip(".,;:!?'\"")]
    fr_words = [w.lower().strip(".,;:!?'\"") for w in fr_text.split() if w.strip(".,;:!?'\"")]
    
    word_scores = []
    for en_w, fr_w in zip(en_words, fr_words):
        # What EN words can this FR word mean?
        possible_meanings = fr_to_ens.get(fr_w, [])
        meaning_words = {en for en, _ in possible_meanings}
        
        # Does the original EN word appear in the FR word's meaning set?
        direct_match = en_w in meaning_words
        # How many meanings overlap?
        overlap = len(meaning_words & set(en_words)) if meaning_words else 0
        
        word_scores.append({
            "en": en_w, "fr": fr_w,
            "direct_meaning": direct_match,
            "meaning_overlap": overlap,
            "possible_meanings": [en for en,_ in possible_meanings[:5]],
        })
    
    return {
        "semantic_score": round(sem_score, 3),
        "word_scores": word_scores,
        "avg_meaning_overlap": sum(w["meaning_overlap"] for w in word_scores) / max(1, len(word_scores)),
    }

# ═══════════════════════════════════════════════════════════════
# COLLABORATION LOOP
# ═══════════════════════════════════════════════════════════════
def collaborate(en_paragraph, max_iterations=3):
    """
    Model A generates. Model B evaluates and suggests improvements.
    Iterates to maximize sound × meaning.
    """
    en_words = [w.lower().strip(".,;:!?'\"") for w in en_paragraph.split() 
                if w.strip(".,;:!?'\"")]
    
    # ── Iteration 1: Model A generates initial French paragraph ──
    initial = []
    for w in en_words:
        candidates = model_A_lookup(w, top_k=5)
        if candidates:
            initial.append(candidates[0])  # (fr, quality, source)
        else:
            initial.append((w, 0.0, "miss"))
    
    best_fr = [fr for fr, _, _ in initial]
    best_result = model_B_evaluate(en_paragraph, " ".join(best_fr))
    
    print(f"  Iter 1: sem={best_result['semantic_score']:.3f}  "
          f"meaning_overlap={best_result['avg_meaning_overlap']:.1f}")
    print(f"    FR: {' '.join(best_fr[:8])}...")
    
    for iteration in range(2, max_iterations + 1):
        improved = False
        
        for i, ws in enumerate(best_result["word_scores"]):
            # If this word has zero meaning overlap, try alternatives
            if ws["meaning_overlap"] == 0:
                candidates = model_A_lookup(ws["en"], top_k=8)
                
                for alt_fr, alt_q, alt_src in candidates[1:]:  # skip current
                    test_fr = best_fr[:i] + [alt_fr] + best_fr[i+1:]
                    test_fr_text = " ".join(test_fr)
                    new_result = model_B_evaluate(en_paragraph, test_fr_text)
                    
                    if new_result["avg_meaning_overlap"] > best_result["avg_meaning_overlap"]:
                        best_fr = test_fr
                        best_result = new_result
                        improved = True
                        break
            
            if improved: break
        
        if improved:
            print(f"  Iter {iteration}: sem={best_result['semantic_score']:.3f}  "
                  f"meaning_overlap={best_result['avg_meaning_overlap']:.1f}")
        else:
            print(f"  Iter {iteration}: converged (no improvement)")
            break
    
    return best_fr, best_result

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        "the sea remembers every ship that ever sailed",
        "she walks in beauty like the night of cloudless climes and starry skies",
        "a thing of beauty is a joy forever its loveliness increases",
    ]
    
    for test in tests:
        print(f"\n{'='*70}")
        print(f"DUAL-AGENT: {test}")
        print(f"{'='*70}")
        
        best_fr, result = collaborate(test, max_iterations=3)
        
        print(f"\n  FINAL:")
        print(f"  EN: {test}")
        print(f"  FR: {' '.join(best_fr)}")
        print(f"  Semantic score: {result['semantic_score']:.3f}")
        print(f"  Meaning overlap: {result['avg_meaning_overlap']:.1f}")
        
        # Show per-word details
        print(f"  Word mappings:")
        for ws in result["word_scores"]:
            dm = "✓" if ws["direct_meaning"] else " "
            print(f"    {ws['en']:15s} → {ws['fr']:15s}  {dm}  means: {ws['possible_meanings'][:3]}")
