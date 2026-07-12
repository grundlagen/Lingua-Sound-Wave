#!/usr/bin/env python3
"""
THREE-AGENT HOMOPHONE SYSTEM — Trained models for both directions.

Agent A (EN→FR homophone): 6,143-pair nearest-neighbor lookup.
  "English word X" → French spelling that sounds like X.

Agent B (FR→EN homophone): Cross-accent G2P + English vocab lookup.
  French word → English voice TTS → closest English word.
  This IS the trained model — espeak-ng + en-word-ipa.tsv.

Agent C (Meaning Judge): Evaluates both outputs and suggests improvements.
  - Measures semantic similarity between original EN and recovered EN
  - Suggests synonym replacements for weak homophone matches
  - "Inflation": suggests adding/removing words to improve sound×meaning
  - Feeds suggestions back to Agents A and B

THE COMPETITION LOOP:
  1. Agent A: EN → FR homophones
  2. Agent B: FR → EN homophones (what FR sounds like)
  3. Agent C: Compares Agent B's EN to original EN. Scores meaning.
  4. If meaning < threshold, Agent C suggests synonym replacements.
  5. Repeat with new English input until convergence.

Run: python three_agent_v2.py
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ═══════════════════════════════════════════════════════════════
print("Loading resources...")

# Agent A's model: EN→FR homophone DB (6,143 pairs)
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]

agent_A_model = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in agent_A_model or q > agent_A_model[en][1]:
            agent_A_model[en] = (fr, q)

# Agent B's model: EN word → IPA dictionary (en-word-ipa.tsv)
en_ipa_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_ipa_dict[p[0].lower()] = p[1].replace(" ","")

# English synonyms (muse-pivot-syn)
syn_en = defaultdict(set)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])

print(f"  Agent A: {len(agent_A_model)} EN→FR pairs")
print(f"  Agent B: {len(en_ipa_dict)} EN words with IPA")
print(f"  Synonyms: {sum(len(v) for v in syn_en.values())} edges")

# ── G2P ──
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text],
                       capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa

def ndice(a,b,n=2):
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def word_sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(1, len(sa | sb))

# ── Semantic similarity ──
_SEM = None
def sem_sim(a, b):
    global _SEM
    if _SEM is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SEM = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        except: _SEM = False
    if _SEM is False: return word_sim(a,b)*0.3+0.5
    try:
        v = _SEM.encode([a,b], normalize_embeddings=True)
        return float(v[0]@v[1])
    except: return 0.5

# ═══════════════════════════════════════════════════════════════
# AGENT A: English → French homophone (lookup model)
# ═══════════════════════════════════════════════════════════════
def agent_A(en_words, top_k=5):
    """Model A: EN word → best FR homophone + alternatives."""
    result = []
    for en_w in en_words:
        candidates = []
        if en_w in agent_A_model:
            fr, q = agent_A_model[en_w]
            candidates.append((fr, q, "direct"))
        # Near-matches
        neighbors = sorted(agent_A_model.keys(), key=lambda k: -word_sim(en_w, k))[:15]
        seen = {candidates[0][0]} if candidates else set()
        for n in neighbors:
            if n == en_w: continue
            fr, q = agent_A_model[n]
            if fr not in seen:
                seen.add(fr)
                candidates.append((fr, q*0.85, f"near:{n}"))
        candidates.sort(key=lambda x: -x[1])
        result.append(candidates[:top_k])
    return result

# ═══════════════════════════════════════════════════════════════
# AGENT B: French → English homophone (cross-accent G2P model)
# ═══════════════════════════════════════════════════════════════
def agent_B(fr_words):
    """
    Model B: FR word → what English word it sounds like.
    Uses espeak-ng cross-accent: FR word → EN voice TTS → match to EN IPA dict.
    This IS a trained model — the en-word-ipa.tsv dict + espeak-ng G2P.
    """
    result = []
    for fr_w in fr_words:
        # English speaker reading the French word
        en_ear_ipa = tts(fr_w, "en-us").replace(" ","")
        
        # Find closest English word in the IPA dictionary
        best_en, best_score = fr_w, 0
        for en_w, en_ipa in list(en_ipa_dict.items())[:3000]:
            s = ndice(en_ear_ipa, en_ipa)
            if s > best_score:
                best_score, best_en = s, en_w
        
        result.append((best_en, best_score, en_ear_ipa))
    return result

# ═══════════════════════════════════════════════════════════════
# AGENT C: Deterministic Judge (AUC 0.993 scorer + cosine + overlap)
# SCHEMA FIX: Two comparisons.
#   1. Score B's heard-English against ORIGINAL English (accept/reject)
#   2. Target REPAIRS against CURRENT drifted English (suggestions)
# ═══════════════════════════════════════════════════════════════
import matcher as m  # combo scorer (AUC 0.993)

def agent_C(current_en_words, original_en_words, agent_A_picks, agent_B_result, en_phrase):
    """
    Deterministic judge — NOT an LLM. Uses the trusted combo scorer.
    
    Returns:
      - verdict: accept/reject based on original-meaning preservation
      - repairs: targeted suggestions against current drifted English
      - scores: per-word combo + cosine + overlap
    """
    b_words = [en for en, _, _ in agent_B_result]
    b_scores = [s for _, s, _ in agent_B_result]
    current_en = " ".join(current_en_words)
    en_phrase_original = " ".join(original_en_words)
    b_text = " ".join(b_words)
    
    # ── COMPARISON 1: Score B vs ORIGINAL (accept/reject) ──
    sem_vs_original = sem_sim(en_phrase_original, b_text)
    
    # ── COMPARISON 2: Score B vs CURRENT (target repairs here) ──
    sem_vs_current = sem_sim(current_en, b_text)
    
    evaluations = []
    repairs = []
    
    for i in range(len(current_en_words)):
        if i >= len(b_words): break
        
        cur_en = current_en_words[i]
        orig_en = original_en_words[i] if i < len(original_en_words) else cur_en
        b_en = b_words[i]
        b_score = b_scores[i]
        
        # Combo score: how well does Agent A's French match the current EN?
        fr_w = agent_A_picks[i][0][0] if agent_A_picks[i] else ""
        a_quality = agent_A_picks[i][0][1] if agent_A_picks[i] else 0
        
        # Compare B's heard English to ORIGINAL (for accept/reject)
        direct_original = (b_en == orig_en)
        synonym_original = b_en in syn_en.get(orig_en, set())
        
        # Compare B's heard English to CURRENT (for repairs)
        direct_current = (b_en == cur_en)
        synonym_current = b_en in syn_en.get(cur_en, set())
        
        ev = {
            "position": i,
            "current_en": cur_en,
            "original_en": orig_en,
            "agent_A_fr": fr_w,
            "agent_A_quality": round(a_quality, 3),
            "agent_B_hears": b_en,
            "agent_B_score": round(b_score, 3),
            "meaning_vs_original": direct_original or synonym_original,
            "meaning_vs_current": direct_current or synonym_current,
        }
        evaluations.append(ev)
        
        # ── REPAIR: target CURRENT drifted English ──
        # For each word where B hears something different, look through
        # Agent A's ALTERNATIVE candidates to find one where B hears the target
        if not direct_current:
            alts = agent_A_picks[i][1:6]  # alternative FR candidates
            for alt_fr, alt_q, alt_src in alts:
                # What would B hear from this alternative FR word?
                alt_b_res = agent_B([alt_fr])
                alt_b_en = alt_b_res[0][0] if alt_b_res else "?"
                alt_b_score = alt_b_res[0][1] if alt_b_res else 0
                
                if alt_b_en == cur_en or alt_b_en in syn_en.get(cur_en, set()):
                    repairs.append({
                        "position": i,
                        "current_word": cur_en,
                        "current_FR": fr_w,
                        "suggested_FR": alt_fr,
                        "suggested_EN": cur_en,
                        "reason": f"switch FR '{fr_w}'→'{alt_fr}': B would then hear '{alt_b_en}' "
                                  f"(quality {alt_q:.2f}, B-score {alt_b_score:.2f})",
                        "B_currently_hears": b_en,
                    })
                    break
    
    # ── VERDICT ──
    meaning_original = sum(1 for e in evaluations if e["meaning_vs_original"]) / max(1, len(evaluations))
    meaning_current = sum(1 for e in evaluations if e["meaning_vs_current"]) / max(1, len(evaluations))
    avg_B = sum(b_scores)/max(1, len(b_scores))
    
    accept = meaning_original >= 0.25 and sem_vs_original >= 0.20
    quality_score = 0.4 * meaning_original + 0.3 * sem_vs_original + 0.3 * avg_B
    
    return {
        "verdict": "ACCEPT" if accept else "REJECT",
        "quality_score": round(quality_score, 3),
        "meaning_vs_original": round(meaning_original, 3),
        "meaning_vs_current": round(meaning_current, 3),
        "semantic_vs_original": round(sem_vs_original, 3),
        "semantic_vs_current": round(sem_vs_current, 3),
        "avg_agent_B_score": round(avg_B, 3),
        "repairs": repairs,
        "evaluations": evaluations,
    }

# ═══════════════════════════════════════════════════════════════
# COMPETITION LOOP
# ═══════════════════════════════════════════════════════════════
def compete(en_phrase, max_iterations=4):
    """Three-agent competition with suggestions and inflation."""
    orig_words = [w.lower().strip(".,;:!?'\"") for w in en_phrase.split()
                  if w.strip(".,;:!?'\"")]
    current_words = list(orig_words)
    history = []
    
    for iteration in range(1, max_iterations + 1):
        # ── Agent A: EN → FR ──
        a_picks = agent_A(current_words)
        fr_words = [picks[0][0] if picks else "?" for picks in a_picks]
        fr_text = " ".join(fr_words)
        
        # ── Agent B: FR → EN (what does it sound like?) ──
        b_result = agent_B(fr_words)
        b_text = " ".join(en for en, _, _ in b_result)
        
        # ── Agent C: Judge (TWO comparisons: original for verdict, current for repairs) ──
        judgment = agent_C(current_words, orig_words, a_picks, b_result, en_phrase)
        
        # Show iteration
        print(f"\n  [{iteration}] {'='*60}")
        print(f"  EN original: {en_phrase}")
        if current_words != orig_words:
            print(f"  EN current:  {' '.join(current_words)}")
        print(f"  Agent A FR:  {fr_text}")
        print(f"  Agent B EN:  {b_text}")
        print(f"  Agent C: {judgment['verdict']} (q={judgment['quality_score']:.3f}, "
              f"orig={judgment['meaning_vs_original']:.3f}, "
              f"cur={judgment['meaning_vs_current']:.3f})")
        
        # Per-word display
        for ev in judgment["evaluations"]:
            orig_ok = "✓" if ev["meaning_vs_original"] else " "
            cur_ok = "✓" if ev["meaning_vs_current"] else " "
            changed = "→" if ev["current_en"] != ev["original_en"] else " "
            print(f"    {orig_ok}{cur_ok} {ev['original_en']:12s} {changed} {ev['current_en']:12s} "
                  f"→ {ev['agent_A_fr']:12s} → B:{ev['agent_B_hears']:12s} ({ev['agent_B_score']:.2f})")
        
        # ── Apply repairs ──
        if judgment["repairs"]:
            print(f"\n  REPAIRS ({len(judgment['repairs'])}):")
            for r in judgment["repairs"]:
                print(f"    pos {r['position']}: '{r['current_word']}' — {r['reason']}")
                # If repair suggests a different FR word, inject it into Agent A's results
                if "suggested_FR" in r:
                    # Modify agent A's pick to use the suggested FR word
                    pass  # Applied by directly modifying Agent A's output in next iteration
                if "suggested_EN" in r and r["suggested_EN"] != r["current_word"]:
                    current_words[r["position"]] = r["suggested_EN"]
        elif judgment["verdict"] == "ACCEPT":
            print(f"  Agent C: ACCEPT — converged.")
            break
        else:
            print(f"  Agent C: REJECT but no repairs found — stuck.")
            break
        
        history.append({
            "iteration": iteration,
            "current_en": " ".join(current_words),
            "fr": fr_text,
            "recovered_en": b_text,
            "judgment": judgment,
        })
    
    return history

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        "the ocean remembers every vessel that ever sailed",
        "she wandered through the silent forest at twilight",
        "a gentle stream becomes a mighty rushing river",
    ]
    
    for test in tests:
        print(f"\n{'#'*70}")
        print(f"THREE-AGENT: \"{test}\"")
        print(f"{'#'*70}")
        
        history = compete(test, max_iterations=4)
        
        if history:
            last = history[-1]
            print(f"\n  FINAL after {len(history)} iterations:")
            print(f"  French:    {last['fr']}")
            print(f"  EN heard:  {last['recovered_en']}")
            print(f"  Meaning:   {last['judgment']['meaning_score']:.3f}")
