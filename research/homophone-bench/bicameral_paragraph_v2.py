#!/usr/bin/env python3
"""
BICAMERAL PARAGRAPH v2 — Sound-stanza clustering with dual meaning.

Instead of EN synonym clusters (too sparse), we group words by their
SHARED FRENCH SOUND. Words that map to the same French homophone class
naturally form a "sound stanza" — they share a sonic identity while
bringing diverse meanings through the homophone siblings.

Each stanza:
  - Has a SOUND THEME (the French sound they all share)
  - Has an ENGLISH MEANING (what the words are about, via synonym clustering)
  - Has a FRENCH MEANING (what the French text says, via homophone sense)
  - Reads as coherent text in both languages

THE FULL MATHEMATICAL LAYER:

  Let H be the set of French homophone classes.
  For each class h ∈ H, define the English preimage:
    E_h = {e ∈ E | ∃f ∈ h with S(e,f) ≥ θ}

  Words in E_h share a sonic identity in French. Within E_h, we find
  semantic clusters via EN synonym edges. Each sub-cluster is a stanza.

  The bicameral paragraph is a sequence of stanzas S₁, S₂, ..., Sₖ where:
    - Each Sᵢ has English words from E_hᵢ (shared French sound)
    - English side reads with cohesion (synonyms, collocations)
    - French side reads naturally (grammatical French, proper liaison)
    - Both sides carry meaning independently

Run: python bicameral_paragraph_v2.py --n 80
"""

from __future__ import annotations

import argparse, os, random
from collections import defaultdict, Counter

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
from topological_flow_v3 import build_polysemy_graph, STOP

FUNC_GLUE = {
    "the":"le","a":"un","an":"un","of":"de","to":"à","in":"en","on":"sur",
    "at":"à","for":"pour","and":"et","or":"ou","is":"est","are":"sont",
    "was":"était","be":"être","it":"il","he":"il","she":"elle","we":"nous",
    "you":"vous","they":"ils","my":"mon","his":"son","her":"sa","our":"notre",
    "this":"ce","that":"ce","not":"pas","but":"mais","so":"donc","as":"comme",
    "by":"par","with":"avec","do":"faire","did":"fait","has":"a","have":"ont",
    "all":"tout","some":"quelques","more":"plus","very":"très","no":"non",
    "from":"de","its":"son","their":"leur","your":"votre","our":"notre",
}

EN_ARTICLES = ["the","a","an","of","in","on","at","to","for","and","or","is","are","was","be","it","he","she","we","they","my","his","her","its","our","their","this","that","not","but","so","as","by","with","from","no","all","some","more","very"]
FR_ARTICLES = ["le","la","les","de","des","du","un","une","à","en","sur","dans","pour","et","ou","est","sont","était","être","il","elle","nous","vous","ils","mon","son","sa","notre","ce","pas","mais","donc","comme","par","avec","tout","quelques","plus","très","non"]

def build_sound_stanzas(n_words=80, min_sound=0.55):
    print("Building algebraic graph...")
    sound, closed_means, homophone, syn_en, syn_fr, bridge = build_polysemy_graph(".")
    
    # Load corpus
    corpus = []
    for i,line in enumerate(open("all_composition_words.tsv",encoding="utf-8")):
        if i==0: continue
        p=line.rstrip("\n").split("\t")
        if len(p)>=2 and p[1] not in STOP and 3<=len(p[1])<=8:
            corpus.append(p[1])
    corpus = corpus[:n_words]
    E = set(corpus)
    
    # ── STAGE 1: Group by SHARED FRENCH SOUND ──
    # For each English word, what French word(s) does it sound like?
    # Group English words that share French sound matches.
    
    fr_to_en_sound = defaultdict(list)  # French word → [(en_word, score), ...]
    for en_w in E:
        for f, s in sound.get(en_w, [])[:5]:
            if s >= min_sound:
                fr_to_en_sound[f].append((en_w, s))
    
    # Build sound stanzas: English words sharing French sounds
    sound_groups = []  # [(fr_sound, [en_words], [fr_variants])]
    used_en = set()
    
    for fr_word, en_list in sorted(fr_to_en_sound.items(), key=lambda x: -len(x[1])):
        en_words = [e for e,s in en_list if e not in used_en]
        if len(en_words) >= 2:
            # Get full homophone class for this sound
            hom_class = homophone.get(fr_word, {fr_word})
            sound_groups.append((fr_word, en_words, sorted(hom_class)[:5]))
            used_en.update(en_words)
    
    # Remaining solo words
    for en_w in E:
        if en_w not in used_en:
            for f, s in sound.get(en_w, [])[:3]:
                if s >= min_sound:
                    sound_groups.append((f, [en_w], [f]))
                    used_en.add(en_w)
                    break
    
    # ── STAGE 2: Build bicameral stanzas ──
    # Each sound group → a stanza with English meaning + French meaning
    
    stanzas = []
    
    for i, (fr_sound, en_words, fr_variants) in enumerate(sound_groups[:15]):
        if not en_words: continue
        
        # ── English side ──
        # Find semantic cohesion among these EN words
        en_syn_links = []
        for w1 in en_words:
            for w2 in en_words:
                if w1 < w2 and w2 in syn_en.get(w1, set()):
                    en_syn_links.append((w1, w2))
        
        # What is this stanza "about" in English?
        en_theme = en_words[0] if en_words else "?"
        if len(en_words) >= 2:
            # Find the shared meaning through synonym expansion
            shared = set(en_words)
            for w in en_words:
                shared &= (syn_en.get(w, set()) | {w})
            if shared:
                en_theme = sorted(shared, key=len)[0]
        
        # Build English text: content words + articles
        en_text_parts = []
        for j, w in enumerate(en_words):
            if j > 0 and j % 4 == 0:
                art = random.choice(EN_ARTICLES[:15])
                en_text_parts.append(art)
            en_text_parts.append(w)
        en_line = " ".join(en_text_parts)
        
        # ── French side ──
        # Map each EN word to its best French match
        fr_text_parts = []
        for j, w in enumerate(en_words):
            best_fr = fr_sound
            best_s = 0
            for f, s in sound.get(w, [])[:5]:
                if s > best_s:
                    best_s = s
                    best_fr = f
            if j > 0 and j % 4 == 0:
                art_en = en_text_parts[-3] if len(en_text_parts) >= 3 else "the"
                art_fr = FUNC_GLUE.get(art_en, art_en)
                fr_text_parts.append(art_fr)
            fr_text_parts.append(best_fr)
        fr_line = " ".join(fr_text_parts)
        
        # ── Meaning labels ──
        # What do the French words mean?
        fr_meanings = set()
        for w in en_words:
            for f, _ in sound.get(w, [])[:3]:
                fr_meanings |= closed_means.get(f, set()) & E
        fr_meaning_sample = sorted(fr_meanings)[:6]
        
        stanzas.append({
            "theme": en_theme,
            "sound": fr_sound,
            "sound_class": fr_variants,
            "en_words": en_words,
            "en_text": en_line,
            "fr_text": fr_line,
            "en_meaning": en_words[:5],
            "fr_meaning": fr_meaning_sample,
            "syn_links": en_syn_links,
        })
    
    # ── OUTPUT ──
    total_en = sum(len(s["en_words"]) for s in stanzas)
    
    print(f"\n{'='*70}")
    print(f"BICAMERAL PARAGRAPH — {len(stanzas)} Sound Stanzas, {total_en} words")
    print(f"{'='*70}")
    
    print(f"\n{'─'*70}")
    print(f"ENGLISH TEXT:")
    print(f"{'─'*70}")
    for s in stanzas:
        title = f"\n  [{s['theme']} — sounds like '{s['sound']}']"
        print(title)
        print(f"  {s['en_text']}")
    
    print(f"\n{'─'*70}")
    print(f"FRENCH TEXT:")
    print(f"{'─'*70}")
    for s in stanzas:
        title = f"\n  [{s['sound']} ≈ {', '.join(s['sound_class'][:4])}]"
        print(title)
        print(f"  {s['fr_text']}")
    
    print(f"\n{'─'*70}")
    print(f"STANZA-BY-STANZA MEANING:")
    print(f"{'─'*70}")
    for i, s in enumerate(stanzas):
        print(f"\n  STANZA {i+1}: [{s['theme']}]")
        print(f"    Sound:     {s['sound']} ≈ {', '.join(s['sound_class'][:4])}")
        print(f"    EN words:  {', '.join(s['en_words'])}")
        print(f"    EN means:  about '{', '.join(s['en_meaning'][:4])}'")
        print(f"    FR means:  covers: {', '.join(s['fr_meaning'][:5])}")
        print(f"    EN text:   {s['en_text']}")
        print(f"    FR text:   {s['fr_text']}")
        if s['syn_links']:
            print(f"    syn links: {', '.join(f'{a}~{b}' for a,b in s['syn_links'][:3])}")
    
    # ── MATHEMATICAL ANALYSIS ──
    import numpy as np
    all_scores = []
    for s in stanzas:
        for w in s["en_words"]:
            best = max((sc for f,sc in sound.get(w,[(0,0)])[:5]), default=0)
            all_scores.append(best)
    
    print(f"\n{'─'*70}")
    print(f"MATHEMATICAL PROPERTIES:")
    print(f"  Stanzas:            {len(stanzas)}")
    print(f"  Total words:        {total_en}")
    print(f"  Sound quality:      μ={np.mean(all_scores):.3f} σ={np.std(all_scores):.3f}")
    print(f"  ≥0.70 sound:        {sum(1 for s in all_scores if s>=0.70)}/{len(all_scores)}")
    
    total_syn = sum(len(s["syn_links"]) for s in stanzas)
    total_pairs = sum(len(s["en_words"])*(len(s["en_words"])-1)//2 for s in stanzas)
    print(f"  Syn density:        {total_syn}/{total_pairs} ({100*total_syn/max(1,total_pairs):.0f}%)")
    print(f"  Avg stanza size:    {total_en/len(stanzas):.1f} words")
    
    # Save
    with open("bicameral_paragraph_v2.txt","w") as f:
        f.write("BICAMERAL PARAGRAPH v2\n=========\n\n")
        f.write("ENGLISH:\n")
        for s in stanzas:
            f.write(f"\n[{s['theme']}]\n{s['en_text']}\n")
        f.write("\n\nFRENCH:\n")
        for s in stanzas:
            f.write(f"\n[{s['sound']} ≈ {', '.join(s['sound_class'][:4])}]\n{s['fr_text']}\n")
        f.write(f"\n\n{total_en} words, {len(stanzas)} stanzas\n")
    
    print(f"\n  Saved to bicameral_paragraph_v2.txt")
    return stanzas

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=80)
    args=ap.parse_args()
    build_sound_stanzas(args.n)

if __name__=="__main__": main()
