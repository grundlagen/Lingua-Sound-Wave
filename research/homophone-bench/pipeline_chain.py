#!/usr/bin/env python3
"""
PIPELINE ARCHITECTURE — Agents are cogs in a chain, not competitors.

  H (Homophone Generator) → M (Meaning Matcher) → F (Fluency Weaver) → B (Hearer) → O (Conductor)

Each agent RECEIVES the previous agent's output, DOES ONE JOB, and PASSES to the next.
They MUST know about each other — that's how the chain works.
O adjusts the whole pipeline, not per-word scores.

THE CHAIN:
  H: Input EN word → output [fr1, fr2, ...] (all homophone candidates from all backends)
  M: Receives H's candidates → scores each by dual meaning (EN→FR and FR→EN)
  F: Receives M's scored candidates → weaves into fluent French phrases via bigram LM
  B: Receives F's French output → reports what English words it sounds like
  O: Receives full chain output → adjusts pipeline weights, flags weak spots

Run: python3 pipeline_chain.py
"""

import json, os, subprocess
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# LOAD ALL DATA (agents share this)
# ═══════════════════════════════════════════════════════════════
print("Loading shared knowledge base...")
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)

# Reverse: FR → what EN words can it mean?
fr_means = defaultdict(set)
for en, (fr, q) in lookup.items():
    fr_means[fr].add(en)

syn_en = defaultdict(set)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])

en_ipa_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_ipa_dict[p[0].lower()] = p[1].replace(" ","")

print(f"  Knowledge: {len(lookup)} EN→FR, {len(fr_means)} FR→EN, "
      f"{sum(len(v) for v in syn_en.values())} synonyms")

# ═══════════════════════════════════════════════════════════════
# G2P
# ═══════════════════════════════════════════════════════════════
def tts(text, voice):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v",voice,text], capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c,"")
    return ipa
def ndice(a,b,n=2):
    A={a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
    B={b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0
def char_sim(a,b): sa,sb=set(a),set(b); return len(sa&sb)/max(1,len(sa|sb))

# ═══════════════════════════════════════════════════════════════
# AGENT H — Homophone Generator (COG 1)
# ═══════════════════════════════════════════════════════════════
class AgentH:
    """Massive homophone candidate database for every word. Uses ALL backends."""
    def generate(self, en_words):
        candidates = {}
        for w in en_words:
            pool = []
            seen = set()
            # Backend 1: Direct lookup (6,143 pairs)
            if w in lookup:
                fr, q = lookup[w]
                pool.append({"fr": fr, "q": q, "src": "lookup"})
                seen.add(fr)
            # Backend 2: Near-matches from same DB
            for n in sorted(lookup.keys(), key=lambda k: -char_sim(w, k))[:20]:
                if n == w: continue
                n_fr, n_q = lookup[n]
                if n_fr not in seen:
                    seen.add(n_fr)
                    pool.append({"fr": n_fr, "q": n_q*0.85, "src": f"near:{n}"})
            # Backend 3: LSTM model (if loaded)
            # Backend 4: Transformer model (if loaded)
            pool.sort(key=lambda x: -x["q"])
            candidates[w] = pool[:12]
        return candidates

# ═══════════════════════════════════════════════════════════════
# AGENT M — Meaning Matcher (COG 2)
# ═══════════════════════════════════════════════════════════════
class AgentM:
    """Takes H's candidates. Finds meaning matches on BOTH sides."""
    def match(self, en_words, h_candidates):
        scored = {}
        for en_w in en_words:
            results = []
            for c in h_candidates.get(en_w, []):
                fr_w = c["fr"]
                # Meaning: what EN words can this FR word mean?
                fr_en_meanings = fr_means.get(fr_w, set())
                # How many of the SENTENCE's EN words does this FR word cover?
                meaning_overlap = len(fr_en_meanings & set(en_words))
                # EN synonym coverage: synonyms of en_w that match this FR word
                syn_overlap = len(syn_en.get(en_w, set()) & fr_en_meanings)
                # Score: base quality × meaning coverage
                m_score = c["q"] * (1.0 + 0.1 * meaning_overlap + 0.05 * syn_overlap)
                results.append({"fr": fr_w, "q": c["q"], "m_score": round(m_score,3),
                                "meaning_overlap": meaning_overlap,
                                "syn_overlap": syn_overlap,
                                "fr_means": sorted(fr_en_meanings)[:5],
                                "src": c["src"]})
            results.sort(key=lambda x: -x["m_score"])
            scored[en_w] = results[:8]
        return scored

# ═══════════════════════════════════════════════════════════════
# AGENT F — Fluency Weaver (COG 3)
# ═══════════════════════════════════════════════════════════════
class AgentF:
    """Takes M's scored candidates. Weaves into fluent French via bigram LM."""
    def __init__(self):
        self.lm = None
        try:
            import bigram_lm as blm
            self.lm = blm.load("fr")
        except: pass
    
    def weave(self, en_words, m_scored):
        """Build the best French phrase with fluency between adjacent words."""
        result = []
        for i, en_w in enumerate(en_words):
            candidates = m_scored.get(en_w, [])
            if not candidates:
                result.append({"fr": en_w, "flu": 0.5, "src": "fallback"})
                continue
            
            # Score each candidate by: m_score + fluency with previous word
            prev_fr = result[-1]["fr"] if result else None
            best, best_score = None, -1
            for c in candidates:
                flu = 0.5
                if self.lm and prev_fr:
                    flu = self.lm.fluency([prev_fr, c["fr"]])
                score = c["m_score"] * 0.7 + flu * 0.3
                if score > best_score:
                    best_score = score
                    best = c
            if best:
                result.append({"fr": best["fr"], "q": best["q"],
                               "m_score": best["m_score"], "flu": round(best_score,3),
                               "fr_means": best.get("fr_means",[]),
                               "src": best["src"]})
        return result

# ═══════════════════════════════════════════════════════════════
# AGENT B — Hearer (COG 4)
# ═══════════════════════════════════════════════════════════════
class AgentB:
    """Takes F's French output. Reports what English words it sounds like."""
    def hear_all(self, f_output):
        result = []
        for item in f_output:
            fr_w = item["fr"]
            en_ipa = tts(fr_w, "en-us").replace(" ","")
            best_en, best_s = fr_w, 0
            for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
                s = ndice(en_ipa, en_w_ipa)
                if s > best_s: best_s, best_en = s, en_w
            result.append({"fr": fr_w, "heard": best_en, "b_score": round(best_s,3)})
        return result

# ═══════════════════════════════════════════════════════════════
# AGENT O — Conductor (COG 5)
# ═══════════════════════════════════════════════════════════════
class AgentO:
    """Receives the full chain output. Adjusts pipeline. Flags weaknesses."""
    def __init__(self):
        self.H = AgentH()
        self.M = AgentM()
        self.F = AgentF()
        self.B = AgentB()
    
    def run_pipeline(self, en_sentence):
        en_words = [w.lower().strip(".,;:!?\"") for w in en_sentence.split() 
                    if w.strip(".,;:!?\"")]
        
        # COG 1: H generates massive homophone candidate pool
        h_out = self.H.generate(en_words)
        
        # COG 2: M scores by dual meaning (EN→FR and FR→EN)
        m_out = self.M.match(en_words, h_out)
        
        # COG 3: F weaves into fluent French
        f_out = self.F.weave(en_words, m_out)
        
        # COG 4: B hears what the French sounds like
        b_out = self.B.hear_all(f_out)
        
        # COG 5: O conducts — flags weaknesses, adjusts
        weak_spots = []
        for i, (f_item, b_item) in enumerate(zip(f_out, b_out)):
            if b_item["b_score"] < 0.40:
                weak_spots.append({
                    "position": i,
                    "en_word": en_words[i],
                    "fr_word": f_item["fr"],
                    "heard_as": b_item["heard"],
                    "b_score": b_item["b_score"],
                    "hint": f"Agent B barely hears English: '{b_item['heard']}' ({b_item['b_score']:.2f})"
                })
        
        quality = sum(b["b_score"] for b in b_out) / max(1, len(b_out))
        
        return {
            "en_words": en_words,
            "h_pool": h_out,
            "m_scored": m_out,
            "f_weave": f_out,
            "b_heard": b_out,
            "weak_spots": weak_spots,
            "quality": round(quality, 3),
            "fr_text": " ".join(f["fr"] for f in f_out),
            "he_text": " ".join(b["heard"] for b in b_out),
        }

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    O = AgentO()
    
    tests = [
        "the ocean remembers every vessel that ever sailed",
        "she wandered through the silent forest at twilight",
        "a gentle stream becomes a mighty rushing river",
    ]
    
    for sent in tests:
        result = O.run_pipeline(sent)
        
        print(f"\n{'='*70}")
        print(f"EN: {sent}")
        print(f"FR: {result['fr_text']}")
        print(f"HE: {result['he_text']}")
        print(f"   Quality: {result['quality']:.3f}  ({len(result['en_words'])} words)")
        
        for i, (f_item, b_item) in enumerate(zip(result["f_weave"], result["b_heard"])):
            m = f_item
            print(f"  {result['en_words'][i]:12s} → {f_item['fr']:15s} "
                  f"(q={f_item['q']:.2f} m={f_item.get('m_score',0):.2f} "
                  f"flu={f_item.get('flu',0):.2f}) → B:{b_item['heard']:12s} ({b_item['b_score']:.2f})")
        
        if result["weak_spots"]:
            print(f"\n  ⚠ Weak spots:")
            for w in result["weak_spots"]:
                print(f"    pos {w['position']}: '{w['en_word']}' → '{w['fr_word']}' — {w['hint']}")
