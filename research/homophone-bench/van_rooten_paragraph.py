#!/usr/bin/env python3
"""
PARAGRAPH-LEVEL VAN ROOTEN — Full Mother Goose stanza homophone carving.

THE CHALLENGE:
  Single-line homophone carving is hard. Paragraph-level is exponentially
  harder because each line's French output must be:
  1. Phonetically close to the English source line (combo score)
  2. Fluent French WITHIN the line (bigram LM coherence)
  3. Fluent French ACROSS lines (inter-line bigram constraint)

APPROACH:
  - Carve each line independently with whole-line beam search
  - Re-rank the top N carves per line using:
    a) combo score (phonetic match)
    b) intra-line bigram coherence (French fluency)
    c) INTER-LINE bigram coherence (transition fluency)
  - Dynamic programming: Viterbi over lines, each state = top-K carve
  - Output: the best sequence of carves that forms a fluent French paragraph

This is the Van Rooten paragraph problem solved as a Viterbi alignment
over a lattice of candidate carves per line.

The bigram LM (bigram-lm-fr.pkl, 195k bigrams from 4-book corpus)
ensures French fluency. The phonetic decoder ensures sound match.
The Viterbi DP ensures inter-line coherence.

RUN: python van_rooten_paragraph.py
"""

import subprocess, os, sys, pickle, math
from collections import defaultdict

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# ── Load the French bigram LM ──
try:
    import bigram_lm as blm
    LM = blm.load("fr")
    print(f"Bigram LM: {LM.N:,} tokens, {len(LM.bigrams):,} bigrams")
except Exception as e:
    print(f"No bigram LM: {e}")
    LM = None

# ── Mother Goose poems (the real ones from Van Rooten / Mots d'Heures) ──
MOTHER_GOOSE = {
    "Humpty Dumpty": [
        "Humpty Dumpty sat on a wall",
        "Humpty Dumpty had a great fall",
        "All the king's horses and all the king's men",
        "Couldn't put Humpty together again",
    ],
    "Little Jack Horner": [
        "Little Jack Horner sat in a corner",
        "Eating a Christmas pie",
        "He put in his thumb and pulled out a plum",
        "And said what a good boy am I",
    ],
    "Mary Had a Little Lamb": [
        "Mary had a little lamb",
        "Its fleece was white as snow",
        "And everywhere that Mary went",
        "The lamb was sure to go",
    ],
    "Hey Diddle Diddle": [
        "Hey diddle diddle the cat and the fiddle",
        "The cow jumped over the moon",
        "The little dog laughed to see such sport",
        "And the dish ran away with the spoon",
    ],
    "Jack and Jill": [
        "Jack and Jill went up the hill",
        "To fetch a pail of water",
        "Jack fell down and broke his crown",
        "And Jill came tumbling after",
    ],
    "Twinkle Twinkle": [
        "Twinkle twinkle little star",
        "How I wonder what you are",
        "Up above the world so high",
        "Like a diamond in the sky",
    ],
}

# ── Clean IPA ──
def clean_ipa(ipa):
    for c in "ˈˌ ": ipa = ipa.replace(c, "")
    return ipa

def en_ipa(text):
    r = subprocess.run(["espeak-ng","-q","--ipa","-v","en-us",text],
                       capture_output=True, text=True, check=True)
    return clean_ipa(r.stdout.strip())

# ── Carve a single line using the existing engine ──
import matcher
import poetry_mode as pm
import whole_line_carve as wlc

def carve_line(line, beam=400):
    """Carve one line using the whole-line engine. Returns list of (combo, coh, fr_text)."""
    root = pm.build_poetry_trie(min_zipf=2.0)
    wlc.force_coverage()
    
    try:
        ipa, nwords, scored = wlc.carve_line(line, root, beam=beam)
        if scored:
            results = []
            for dual, combo, coh, cov, fr, nfr, nf in scored[:30]:
                results.append((combo, coh, fr))
            return results
    except Exception:
        pass
    return []

# ── Bigram fluency ──
def bigram_fluency(text, lm=None):
    """Score French text for bigram coherence. 0-1 score."""
    if lm is None:
        return 0.5  # fallback
    words = [w.lower().strip(".,;:!?'\"") for w in text.split() if w.strip(".,;:!?'\"")]
    if len(words) < 2:
        return 0.5
    return lm.fluency(words)

def inter_line_fluency(text_a, text_b, lm=None):
    """Score bigram from last word of line A to first word of line B."""
    if lm is None:
        return 0.5
    wa = [w.lower().strip(".,;:!?'\"") for w in text_a.split() if w.strip(".,;:!?'\"")]
    wb = [w.lower().strip(".,;:!?'\"") for w in text_b.split() if w.strip(".,;:!?'\"")]
    if not wa or not wb:
        return 0.5
    return lm.fluency([wa[-1], wb[0]])  # just the boundary bigram

# ═══════════════════════════════════════════════════════════════
# VITERBI PARAGRAPH CARVER
# ═══════════════════════════════════════════════════════════════
def viterbi_paragraph(lines, lm=None, top_k=5):
    """
    Dynamic programming over lines.
    
    State: (line_idx, carve_idx) with score:
      score = combo + alpha*bigram_fluency + beta*inter_line_fluency
    
    Transitions: from best carves of line N to best carves of line N+1.
    """
    # Step 1: Carve each line, keep top_k candidates
    line_carves = []
    for i, line in enumerate(lines):
        carves = carve_line(line)
        if not carves:
            print(f"  Line {i}: '{line[:30]}...' — NO CARVE")
            return None
        # Score each carve: combo + fluency
        scored = []
        for combo, coh, fr in carves:
            flu = bigram_fluency(fr, lm) if lm else coh
            score = combo + 0.3 * flu
            scored.append((score, combo, coh, flu, fr))
        scored.sort(reverse=True)
        line_carves.append(scored[:top_k])
        print(f"  Line {i}: '{line[:40]}...' → {len(scored[:top_k])} carves, "
              f"best: {scored[0][4][:50]} (s={scored[0][1]:.2f})")

    # Step 2: Viterbi DP
    # dp[i][j] = (best_score, prev_idx) for line i, carve j
    alpha = 0.3  # weight for intra-line fluency
    beta = 0.2   # weight for inter-line fluency
    
    dp = []
    for i in range(len(line_carves)):
        dp.append([(0.0, -1) for _ in range(len(line_carves[i]))])
    
    # Line 0: score = combo + alpha*fluency
    for j, (score, combo, coh, flu, fr) in enumerate(line_carves[0]):
        dp[0][j] = (score, -1)
    
    # Lines 1..N
    for i in range(1, len(line_carves)):
        for j, (s_j, combo_j, coh_j, flu_j, fr_j) in enumerate(line_carves[i]):
            best_score = -float('inf')
            best_prev = -1
            for k, (s_k, combo_k, coh_k, flu_k, fr_k) in enumerate(line_carves[i-1]):
                inter = inter_line_fluency(fr_k, fr_j, lm)
                score = dp[i-1][k][0] + s_j + beta * inter
                if score > best_score:
                    best_score = score
                    best_prev = k
            dp[i][j] = (best_score, best_prev)
    
    # Step 3: Backtrack
    last_i = len(line_carves) - 1
    best_j = max(range(len(dp[last_i])), key=lambda j: dp[last_i][j][0])
    
    path = []
    i, j = last_i, best_j
    while i >= 0:
        path.append(line_carves[i][j][4])  # French text
        j = dp[i][j][1]
        i -= 1
    path.reverse()
    
    return path

# ═══════════════════════════════════════════════════════════════
print("PARAGRAPH-LEVEL VAN ROOTEN — Viterbi over Mother Goose stanzas")
print("=" * 65)

# Use Humpty Dumpty as the test case
poem_name = "Humpty Dumpty"
lines = MOTHER_GOOSE[poem_name]

print(f"\nPoem: {poem_name} ({len(lines)} lines)")
for i, line in enumerate(lines):
    print(f"  [{i}] {line}")

print(f"\nCarving...")
path = viterbi_paragraph(lines, LM, top_k=5)

if path:
    print(f"\n{'='*65}")
    print(f"VAN ROOTEN PARAGRAPH: {poem_name}")
    print(f"{'='*65}")
    for i, (en, fr) in enumerate(zip(lines, path)):
        print(f"\n  EN: {en}")
        print(f"  FR: {fr}")
        # Score it
        en_i = en_ipa(en)
        fr_i = clean_ipa(subprocess.run(
            ["espeak-ng","-q","--ipa","-v","fr",fr],
            capture_output=True, text=True, check=True).stdout.strip())
        import numpy as np
        # Simple ngram-dice
        def ndice(a,b,n=2):
            A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a)>=n else {a}
            B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b)>=n else {b}
            return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0
        s = ndice(en_i, fr_i)
        print(f"  SOUND: {s:.3f}  FR IPA: [{fr_i[:50]}]")
    
    # Whole paragraph fluency
    full = " ".join(path)
    full_flu = bigram_fluency(full, LM)
    print(f"\n  Full paragraph fluency: {full_flu:.3f}")
    print(f"  Full French: {full}")
else:
    print("\n  FAILED — could not carve all lines.")
