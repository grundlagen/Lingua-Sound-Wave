#!/usr/bin/env python3
"""Van Rooten replication — aggressive filler mode, bigram from corpus-carves."""
import subprocess, matcher, poetry_mode as pm
import phonetic_decoder as pd
import whole_line_carve as wlc
from lexicon_g2p import clean_ipa, load_fr
from collections import Counter
import math

# ── Build a simple French bigram LM from available data ──
def build_simple_fr_lm():
    """Build bigram counts from fr-units, fr-word-ipa, and corpus-carves."""
    uni = Counter()
    bi = Counter()
    ctx = Counter()
    
    # From fr-word-ipa (adjacent words in dictionary = weak signal)
    words = []
    for i,line in enumerate(open("fr-word-ipa.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2 and "(en)" not in p[0]:
            w = p[0].lower().strip("'")
            words.append(w)
    for w in words:
        uni[w] += 1
    for a,b in zip(words, words[1:]):
        bi[(a,b)] += 1
        ctx[a] += 1
    
    # From fr-units (elision/liaison pairs)
    for i,line in enumerate(open("fr-units.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=1 and "'" in p[0]:
            parts = p[0].split("'")
            if len(parts)==2:
                a, b = parts[0]+"'", parts[1]
                bi[(a,b)] += 20; uni[a]+=20; uni[b]+=20; ctx[a]+=20
    
    # From corpus-carves (actual French phrases)
    for i,line in enumerate(open("corpus-carves.tsv",encoding="utf-8")):
        if i==0: continue
        p = line.rstrip("\n").split("\t")
        if len(p)>=2:
            toks = [t.lower().strip("',.;:!?") for t in p[1].split() if t]
            for t in toks: uni[t] += 5
            for a,b in zip(toks, toks[1:]):
                bi[(a,b)] += 5; ctx[a] += 5
    
    # Common French patterns (article + noun, preposition + noun, etc.)
    patterns = [
        ("le",""),("la",""),("les",""),("un",""),("une",""),("de",""),("des",""),
        ("du",""),("à",""),("et",""),("ou",""),("en",""),("sur",""),("dans",""),
        ("pour",""),("avec",""),("sans",""),("chez",""),("par",""),
        ("ne",""),("se",""),("me",""),("te",""),("nous",""),("vous",""),
        ("est",""),("sont",""),("était",""),("être",""),("fait",""),
        ("tout",""),("tous",""),("très",""),("bien",""),("plus",""),
        ("l'",""),("d'",""),("s'",""),("n'",""),("qu'",""),("j'",""),
    ]
    for a,_ in patterns:
        for b in words[:200]:
            bi[(a,b)] += 1; ctx[a] += 1
    
    N = sum(uni.values())
    
    def cond(a,b):
        if ctx.get(a) and bi.get((a,b)):
            return bi[(a,b)] / ctx[a]
        return 0.001 * (uni.get(b,1) / max(1,N))
    
    def fluency(word_list):
        if not word_list: return 0.0
        lp = 0.0
        lp += math.log(max(1e-8, uni.get(word_list[0],1)/max(1,N)))
        for a,b in zip(word_list, word_list[1:]):
            lp += math.log(max(1e-8, cond(a,b)))
        avg_lp = lp / max(1, len(word_list))
        return max(0.0, min(1.0, (avg_lp + 15.0) / 11.0))
    
    return fluency, uni, bi

def en_ipa(t):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",t],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())

print("Building resources...")
fluency_fn, uni, bi = build_simple_fr_lm()
print(f"  LM: {len(uni):,} unigrams, {len(bi):,} bigrams")

root = pm.build_poetry_trie(min_zipf=2.0)
wlc.force_coverage()

lines = [
    "Humpty Dumpty",
    "Humpty Dumpty sat on a wall",
    "Hickory dickory dock",
    "Jack and Jill went up the hill",
    "Mary had a little lamb",
    "Little Jack Horner sat in a corner",
    "Rain rain go away",
    "twinkle twinkle little star",
    "London bridge is falling down",
    "the cat and the fiddle",
]

print("\n" + "="*70)
print("VAN ROOTEN REPLICATION — Aggressive Filler Mode")
print("="*70)

for line in lines:
    print(f"\n{'─'*55}")
    print(f"EN:  {line}")

    # Try multiple parameter settings
    best = None
    for filler_bonus in (1.0, 1.5):
        for word_penalty in (0.0, 0.02):
            pd.WORD_PENALTY = word_penalty
            pd.MIN_WORD_SEGS = 1

            for scale in (1.0, 1.6, 2.2):
                matcher.CHEAP_GAP["h"] = 0.08 * scale
                matcher.GAP = min(0.95, 0.42 * scale)

                try:
                    ipa, nwords, scored = wlc.carve_line(line, root, beam=400)
                    if scored:
                        for dual, combo, coh, cov, fr, nfr, nf in scored[:5]:
                            # Re-score with our fluency
                            toks = [t.lower().strip("'") for t in fr.replace("'"," ").split() if t]
                            flu = fluency_fn(toks)
                            # Filler bonus
                            filler_count = sum(1 for t in toks if t in pm.FILLER)
                            score = combo * (0.3 + 0.7 * flu) + filler_bonus * 0.02 * filler_count
                            if best is None or score > best[0]:
                                best = (score, combo, flu, cov, fr, filler_count)
                except Exception:
                    pass

    if best:
        score, combo, flu, cov, fr, nf = best
        print(f"FR:  {fr}")
        print(f"     s={combo:.3f}  flu={flu:.3f}  cov={cov:.0%}  fillers={nf}")

        # Show top 3 alternatives
        pd.WORD_PENALTY = 0.0
        matcher.CHEAP_GAP["h"] = 0.08
        matcher.GAP = 0.42
        try:
            _, _, scored = wlc.carve_line(line, root, beam=400)
            if scored:
                print(f"     alt:")
                for dual, combo, coh, cov, fr2, nfr, nf2 in scored[:3]:
                    toks2 = [t.lower() for t in fr2.replace("'"," ").split() if t]
                    flu2 = fluency_fn(toks2)
                    fill2 = sum(1 for t in toks2 if t in pm.FILLER)
                    print(f"       {fr2:50s} s={combo:.3f} flu={flu2:.3f} fill={fill2}")
        except:
            pass
    else:
        print("     (no carve)")
