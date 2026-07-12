#!/usr/bin/env python3
"""
DUAL-LANGUAGE PIPELINE — Whatever happens to English must also happen to French.

H generates candidates for BOTH sides (EN→FR and FR→EN).
M scores meaning on BOTH sides (EN synonyms, FR meanings, dual coverage).
F weaves both languages into fluent parallel text.
B hears from BOTH ears (EN ear hears FR, FR ear hears EN).
O conducts the full dual-language orchestra.

Each cog knows what the previous cog produced — they're a chain, not competitors.

Run: python3 dual_pipeline.py
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
# SHARED DATA
# ═══════════════════════════════════════════════════════════════
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)

# Reverse index: FR → what EN words map to it
fr_to_en = defaultdict(set)
for en, (fr, q) in lookup.items():
    fr_to_en[fr].add(en)

# Chain-web: transitive EN↔FR edges (optional)
chain_en_fr = defaultdict(lambda: defaultdict(list))
try:
    for i,line in enumerate(open("chain-web/archive/chain-web-full-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=5 and ":" in p[0] and ":" in p[1]:
            sl,sw = p[0].split(":",1); tl,tw = p[1].split(":",1)
            if sl=="en" and tl=="fr": chain_en_fr[sw][tw].append((int(p[2]), float(p[3])))
            elif sl=="fr" and tl=="en": chain_en_fr[tw][sw].append((int(p[2]), float(p[3])))
except FileNotFoundError: pass

# Loop-certified pairs (optional)
loops = set()
try:
    for i,line in enumerate(open("chain-web/archive/loop-certified-pairs-v7u.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2: loops.add((p[0], p[1])); loops.add((p[1], p[0]))
except FileNotFoundError: pass

# Synonyms
syn_en = defaultdict(set)
syn_fr = defaultdict(set)
for line in open("muse-pivot-syn.tsv",encoding="utf-8"):
    a,b,_ = line.rstrip("\n").split("\t")
    if a.startswith("en:") and b.startswith("en:"):
        syn_en[a[3:]].add(b[3:]); syn_en[b[3:]].add(a[3:])
    elif a.startswith("fr:") and b.startswith("fr:"):
        syn_fr[a[3:]].add(b[3:]); syn_fr[b[3:]].add(a[3:])

print(f"Loaded: {len(lookup)} EN→FR, {len(fr_to_en)} FR→EN, "
      f"{len(loops)} loops, {sum(len(v) for v in chain_en_fr.values())} chain-edges")

# ═══════════════════════════════════════════════════════════════
# G2P
# ═══════════════════════════════════════════════════════════════
en_ipa_dict = {}
for i,line in enumerate(open("en-word-ipa.tsv",encoding="utf-8")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p)>=2 and p[1] and "(fr)" not in p[0]:
        en_ipa_dict[p[0].lower()] = p[1].replace(" ","")

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
# AGENT H — Dual Homophone Generator
# ═══════════════════════════════════════════════════════════════
class AgentH:
    """Generates homophone candidates for BOTH English and French sides."""
    def generate(self, en_words):
        result = {"en_to_fr": {}, "fr_to_en": {}}
        for w in en_words:
            pool = []
            seen = set()
            # EN→FR candidates
            if w in lookup:
                fr, q = lookup[w]
                pool.append({"fr": fr, "q": q, "src": "lookup",
                             "loop": (w,fr) in loops, "chain": w in chain_en_fr and fr in chain_en_fr[w]})
                seen.add(fr)
            for n in sorted(lookup.keys(), key=lambda k: -char_sim(w,k))[:15]:
                if n == w: continue
                n_fr, n_q = lookup[n]
                if n_fr not in seen:
                    seen.add(n_fr)
                    pool.append({"fr": n_fr, "q": n_q*0.85, "src": f"near:{n}"})
            pool.sort(key=lambda x: -x["q"])
            result["en_to_fr"][w] = pool[:10]
            
            # FR→EN candidates (what EN words does the best FR map to?)
            if pool:
                top_fr = pool[0]["fr"]
                result["fr_to_en"][top_fr] = sorted(fr_to_en.get(top_fr, set()))
        return result

# ═══════════════════════════════════════════════════════════════
# AGENT M — Dual Meaning Matcher
# ═══════════════════════════════════════════════════════════════
class AgentM:
    """Scores meaning on BOTH sides: EN synonyms + FR meanings + chain support."""
    def match(self, en_words, h_output):
        en_candidates = h_output["en_to_fr"]
        scored = {}
        for en_w in en_words:
            results = []
            for c in en_candidates.get(en_w, []):
                fr_w = c["fr"]
                # EN side: synonyms of en_w that this FR word covers
                en_syns = syn_en.get(en_w, set())
                fr_meanings = fr_to_en.get(fr_w, set())
                en_side = len(en_syns & fr_meanings)
                # FR side: FR synonyms that map back to en_w
                fr_syns = syn_fr.get(fr_w, set())
                fr_side = sum(1 for fs in fr_syns if en_w in fr_to_en.get(fs, set()))
                # Chain support
                chain_score = 0.05 if c.get("chain") else 0
                loop_score = 0.10 if c.get("loop") else 0
                # Dual meaning score
                m_score = c["q"] * (1.0 + 0.1*(en_side + fr_side) + chain_score + loop_score)
                results.append({"fr": fr_w, "q": c["q"], "m_score": round(m_score,3),
                                "en_side": en_side, "fr_side": fr_side,
                                "loop": c.get("loop",False), "chain": c.get("chain",False)})
            results.sort(key=lambda x: -x["m_score"])
            scored[en_w] = results[:8]
        return scored

# ═══════════════════════════════════════════════════════════════
# AGENT F — Dual Fluency Weaver
# ═══════════════════════════════════════════════════════════════
class AgentF:
    """Weaves into fluent French, tracking fluency for both languages."""
    def weave(self, en_words, m_scored):
        result = []
        for i, en_w in enumerate(en_words):
            candidates = m_scored.get(en_w, [])
            if not candidates:
                result.append({"fr": en_w, "flu": 0.5, "m_score": 0.5, "src": "fallback"})
                continue
            prev_fr = result[-1]["fr"] if result else None
            best, best_score = None, -1
            for c in candidates:
                # Simple fluency: prefer candidates that share characters with neighbors
                flu = 0.5
                if prev_fr:
                    flu = char_sim(prev_fr, c["fr"])
                score = c["m_score"] * 0.6 + flu * 0.4
                if score > best_score:
                    best_score, best = score, c
            if best:
                result.append({"fr": best["fr"], "q": best["q"],
                               "m_score": best["m_score"],
                               "en_side": best.get("en_side",0),
                               "fr_side": best.get("fr_side",0),
                               "loop": best.get("loop",False),
                               "flu": round(best_score,3)})
        return result

# ═══════════════════════════════════════════════════════════════
# AGENT B — Dual Ear Hearer
# ═══════════════════════════════════════════════════════════════
class AgentB:
    """Hears from BOTH ears: EN ear hears FR, FR ear hears EN."""
    def hear_all(self, f_output, en_words):
        result = []
        for item, en_w in zip(f_output, en_words):
            fr_w = item["fr"]
            # EN ear: what English word does this FR sound like?
            en_ipa = tts(fr_w, "en-us").replace(" ","")
            best_en, best_s = fr_w, 0
            for ew, ew_ipa in list(en_ipa_dict.items())[:3000]:
                s = ndice(en_ipa, ew_ipa)
                if s > best_s: best_s, best_en = s, ew
            # FR ear: what does the EN word sound like in French? (reverse ear)
            fr_ipa = tts(en_w, "fr").replace(" ","")
            best_fr, best_fs = en_w, 0
            for fw in sorted(fr_to_en.keys(), key=lambda k: -char_sim(en_w,k))[:500]:
                fw_ipa = tts(fw, "fr").replace(" ","")
                s = ndice(fr_ipa, fw_ipa)
                if s > best_fs: best_fs, best_fr = s, fw
            result.append({"fr": fr_w, "en_heard": best_en, "en_score": round(best_s,3),
                           "en_word": en_w, "fr_heard": best_fr, "fr_score": round(best_fs,3)})
        return result

# ═══════════════════════════════════════════════════════════════
# AGENT O — Dual Conductor
# ═══════════════════════════════════════════════════════════════
class AgentO:
    def __init__(self):
        self.H = AgentH(); self.M = AgentM(); self.F = AgentF(); self.B = AgentB()
    
    def run_pipeline(self, en_sentence):
        en_words = [w.lower().strip(".,;:!?\"") for w in en_sentence.split() 
                    if w.strip(".,;:!?\"")]
        
        h_out = self.H.generate(en_words)
        m_out = self.M.match(en_words, h_out)
        f_out = self.F.weave(en_words, m_out)
        b_out = self.B.hear_all(f_out, en_words)
        
        quality = sum(b["en_score"] for b in b_out) / max(1, len(b_out))
        loops_used = sum(1 for f in f_out if f.get("loop"))
        
        return {
            "en_words": en_words, "fr_text": " ".join(f["fr"] for f in f_out),
            "he_text": " ".join(b["en_heard"] for b in b_out),
            "fr_ear_text": " ".join(b["fr_heard"] for b in b_out),
            "quality": round(quality, 3), "loops": loops_used,
            "f_weave": f_out, "b_heard": b_out,
        }

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    O = AgentO()
    for sent in [
        "the ocean remembers every vessel that ever sailed",
        "she wandered through the silent forest at twilight",
        "a gentle stream becomes a mighty rushing river",
    ]:
        r = O.run_pipeline(sent)
        print(f"\n{'='*65}")
        print(f"EN:   {sent}")
        print(f"FR:   {r['fr_text']}")
        print(f"EN👂: {r['he_text']}")
        print(f"FR👂: {r['fr_ear_text']}")
        print(f"  Q={r['quality']:.3f}  loops={r['loops']}  words={len(r['en_words'])}")
        for f,b in zip(r["f_weave"], r["b_heard"]):
            loop = "↺" if f.get("loop") else " "
            en_s = f.get("en_side",0); fr_s = f.get("fr_side",0)
            print(f"  {loop}{b['en_word']:12s}→{b['fr']:15s} "
                  f"EN:{b['en_heard']:12s}({b['en_score']:.2f}) "
                  f"FR:{b['fr_heard']:12s}({b['fr_score']:.2f}) "
                  f"m={f['m_score']:.2f} en={en_s} fr={fr_s}")
