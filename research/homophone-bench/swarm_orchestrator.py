#!/usr/bin/env python3
"""
SWARM ARCHITECTURE — Five independent agents, one orchestrator.

AGENT H (Homophone): Maximizes English-ear match. Uses all 4 backends.
AGENT M (Meaning):     Maximizes semantic preservation. Uses synonym graph.
AGENT F (Fluency):     Maximizes French fluency. Uses bigram LM.
AGENT B (Hearer):      Reports what English word the French sounds like.
AGENT O (Orchestrator):Balances H/M/F proposals. Picks best compromise.

Each agent has ONE job. They don't know about each other.
The orchestrator receives proposals from H, M, and F for each word,
scores them with a weighted sum, and outputs the best French paragraph.

ARCHITECTURE:
  For each EN word:
    H proposes → [(fr_h1, score), (fr_h2, score), ...]
    M proposes → [(fr_m1, score), (fr_m2, score), ...]
    F proposes → [(fr_f1, score), (fr_f2, score), ...]
    B evaluates → what does each proposal sound like?
    O decides → argmax(w_h*H + w_m*M + w_f*F) across all proposals

Run: python3 swarm_orchestrator.py
"""

import json, os, subprocess, sys
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ═══════════════════════════════════════════════════════════════
# SHARED DATA (all agents read, none write)
# ═══════════════════════════════════════════════════════════════
print("Loading shared data...")
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]: lookup[en] = (fr, q)

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

print(f"  Lookup: {len(lookup)} pairs, Synonyms: {sum(len(v) for v in syn_en.values())} edges")

# ═══════════════════════════════════════════════════════════════
# G2P (Agent B's foundation)
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

def char_sim(a,b):
    sa,sb=set(a),set(b); return len(sa&sb)/max(1,len(sa|sb))

# ═══════════════════════════════════════════════════════════════
# AGENT H — Homophone Maximizer
# ═══════════════════════════════════════════════════════════════
class HomophoneAgent:
    """Only cares about: does this French word sound like the English word?"""
    def propose(self, en_word, top_k=5):
        proposals = []
        seen = set()
        # Backend 1: Direct lookup
        if en_word in lookup:
            fr, q = lookup[en_word]
            proposals.append(("H:lookup", fr, q))
            seen.add(fr)
        # Backend 2: Near-matches by character similarity
        neighbors = sorted(lookup.keys(), key=lambda k: -char_sim(en_word, k))[:15]
        for n in neighbors:
            if n == en_word: continue
            fr, q = lookup[n]
            if fr not in seen:
                seen.add(fr)
                proposals.append((f"H:near:{n}", fr, q * 0.85))
        proposals.sort(key=lambda x: -x[2])
        return proposals[:top_k]

# ═══════════════════════════════════════════════════════════════
# AGENT M — Meaning Preserver
# ═══════════════════════════════════════════════════════════════
class MeaningAgent:
    """Only cares about: does this French word preserve the English meaning?"""
    def propose(self, en_word, top_k=5):
        proposals = []
        seen = set()
        # Direct lookup first
        if en_word in lookup:
            fr, q = lookup[en_word]
            proposals.append(("M:direct", fr, q * 0.9))
            seen.add(fr)
        # Synonym-based: find synonyms that have better DB matches
        if en_word in syn_en:
            for syn in syn_en[en_word]:
                if syn in lookup:
                    fr, q = lookup[syn]
                    if fr not in seen:
                        seen.add(fr)
                        proposals.append((f"M:syn:{syn}", fr, q * 0.95))
        # If nothing, fall back to homophone
        if not proposals:
            neighbors = sorted(lookup.keys(), key=lambda k: -char_sim(en_word, k))[:5]
            for n in neighbors:
                if n == en_word: continue
                fr, q = lookup[n]
                if fr not in seen:
                    seen.add(fr)
                    proposals.append((f"M:fallback:{n}", fr, q * 0.7))
        proposals.sort(key=lambda x: -x[2])
        return proposals[:top_k]

# ═══════════════════════════════════════════════════════════════
# AGENT F — Fluency Maximizer
# ═══════════════════════════════════════════════════════════════
class FluencyAgent:
    """Only cares about: does this French word flow with its neighbors?"""
    def __init__(self):
        self.lm = None
        try:
            import bigram_lm as blm
            self.lm = blm.load("fr")
        except: pass
    
    def score(self, fr_word, prev_word=None):
        if self.lm is None: return 0.5
        if prev_word:
            return self.lm.fluency([prev_word, fr_word])
        return 0.5
    
    def propose(self, en_word, prev_word=None, top_k=5):
        # Get all possible FR candidates
        candidates = []
        if en_word in lookup:
            candidates.append(lookup[en_word][0])
        for n in sorted(lookup.keys(), key=lambda k: -char_sim(en_word, k))[:10]:
            if n != en_word: candidates.append(lookup[n][0])
        # Score by fluency + sound
        scored = []
        seen = set()
        for fr in candidates:
            if fr in seen: continue
            seen.add(fr)
            flu = self.score(fr, prev_word)
            sound = char_sim(en_word, fr)  # rough proxy
            scored.append((f"F:flow", fr, 0.5*sound + 0.5*flu))
        scored.sort(key=lambda x: -x[2])
        return scored[:top_k]

# ═══════════════════════════════════════════════════════════════
# AGENT B — English Ear Hearer
# ═══════════════════════════════════════════════════════════════
class HearerAgent:
    """Reports: what English word does this French word sound like?"""
    def hear(self, fr_word):
        en_ipa = tts(fr_word, "en-us").replace(" ","")
        best_en, best_s = fr_word, 0
        for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
            s = ndice(en_ipa, en_w_ipa)
            if s > best_s: best_s, best_en = s, en_w
        return best_en, best_s

# ═══════════════════════════════════════════════════════════════
# AGENT O — Orchestrator
# ═══════════════════════════════════════════════════════════════
class Orchestrator:
    """
    Receives proposals from H, M, F for each word.
    Evaluates each proposal with Agent B (hearing).
    Picks the one that maximizes: w_h * H_score + w_m * M_score + w_f * F_score.
    """
    def __init__(self, w_h=0.5, w_m=0.3, w_f=0.2):
        self.H = HomophoneAgent()
        self.M = MeaningAgent()
        self.F = FluencyAgent()
        self.B = HearerAgent()
        self.w = (w_h, w_m, w_f)
    
    def _score_proposal(self, p, en_word):
        """Score a single proposal: H_score from agent, M_score from agent, B_score from hearer."""
        src, fr, agent_score = p
        # Agent B: what does this FR word sound like?
        heard, b_score = self.B.hear(fr)
        # H-score: how well does Agent B hear the ORIGINAL English word?
        h_score = 1.0 if heard == en_word else (0.8 if heard in syn_en.get(en_word, set()) else b_score)
        # M-score: how meaning-preserving is this?
        m_score = 1.0 if en_word in lookup and fr == lookup[en_word][0] else 0.7
        if "syn" in src: m_score = 0.9  # synonym-based gets boost
        # F-score: fluency (from agent)
        f_score = 1.0 if "F:" in str(src) else 0.5
        return self.w[0]*h_score + self.w[1]*m_score + self.w[2]*f_score, heard, fr
    
    def decide(self, en_word, prev_fr=None):
        """Get proposals from all agents, score them, pick best."""
        h_props = self.H.propose(en_word, top_k=4)
        m_props = self.M.propose(en_word, top_k=4)
        f_props = self.F.propose(en_word, prev_fr, top_k=3)
        
        all_props = h_props + m_props + f_props
        best_score, best_heard, best_fr = -1, "?", "?"
        
        for p in all_props:
            score, heard, fr = self._score_proposal(p, en_word)
            if score > best_score:
                best_score, best_heard, best_fr = score, heard, fr
        
        return best_fr, best_heard, best_score

# ═══════════════════════════════════════════════════════════════
# MAIN — Swarm Competition
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    O = Orchestrator(w_h=0.5, w_m=0.3, w_f=0.2)
    
    sentences = [
        "the ocean remembers every vessel that ever sailed",
        "she wandered through the silent forest at twilight",
        "a gentle stream becomes a mighty rushing river",
    ]
    
    for sent in sentences:
        words = [w.lower().strip(".,;:!?\"") for w in sent.split() if w.strip(".,;:!?\"")]
        results = []
        prev_fr = None
        
        for w in words:
            fr, heard, score = O.decide(w, prev_fr)
            results.append((w, fr, heard, score))
            prev_fr = fr
        
        fr_text = " ".join(fr for _,fr,_,_ in results)
        he_text = " ".join(h for _,_,h,_ in results)
        
        print(f"\n{'='*60}")
        print(f"EN: {sent}")
        print(f"FR: {fr_text}")
        print(f"HE: {he_text}")
        for w,fr,heard,score in results:
            print(f"  {w:12s} → {fr:15s} → B:{heard:12s} (swarm={score:.3f})")
