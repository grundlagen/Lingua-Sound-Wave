"""Refined homophone dictionary (v3): lexicon pronunciations + pair bank.

v3 on top of v2: G2P now comes from curated dictionaries first — Lexique 3
for French (241k words) and WikiPron for English (65k words, UK accent kept
alongside espeak US as a second legitimate variant) — recovered from the
user's prior homophone project (checkcehck repo). espeak remains the
fallback. The matcher gained an EN<->FR equivalence-cost layer and cheap
deletions (offglides, schwa, /h/) ported from the production phoneme.ts and
the user's 2024 notebook rules. The user's 30k-pair historical pair bank is
scored and merged (provenance-tagged). Liaison variants are tried when
concatenating multi-word French phrases.

v2 notes below:

Improvements over build_dictionary_full.py, all aimed at "more good entries,
no poor ones":

  1. G2P overrides — espeak language-switches on ~218 French lexicon words.
     Native French words it butchers (menthe, rythme, dos, vécut...) get
     hand-curated correct IPA; genuine English loanwords (cool, football)
     are tagged `loanword` and EXCLUDED from S/A — matching "design" to
     "designed" is not a discovery. Stray proper nouns are dropped.
  2. Multi-word French matches — for English words without a gold single
     match, search 2-word French phrases ("mayday" ~ "m'aider" style).
     Both component words must be common (zipf >= 3.0) so phrases stay
     natural; the score bar is the same as single words. Tagged `multiword`.
  3. Cognate tagging via the MUSE EN-FR bilingual dictionary: entries where
     the pair also shares meaning get `cognate: true` (the shared
     sound-AND-meaning lexicon, extracted for free).
  4. Alignment decomposition — every kept entry gets the NW alignment
     chunks ("s:s ɛ:ɛ t:t"), aligned-segment coverage, and for multiword
     entries the position of the word boundary (the "pause") relative to
     the English segments.
  5. Tier hygiene — B-tier keeps only the best candidate per English word
     and requires at least one shared exact bigram, killing the
     pure-coincidence tail.

Run: python refine_dictionary.py [n_en=12000] [n_fr=25000]
Outputs: dictionary-v2.json, dictionary-v2.tsv
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from wordfreq import top_n_list, zipf_frequency

import matcher
from matcher import _canonical, _variants, _ngram_channel
from lexicon_g2p import load_en as load_lex_en, load_fr as load_lex_fr, clean_ipa

matcher._vecs = lru_cache(maxsize=None)(matcher._vecs.__wrapped__)
matcher._segs = lru_cache(maxsize=None)(matcher._segs.__wrapped__)

S, A, B = 0.90, 0.78, 0.62
SHORTLIST = 15
KEEP_PER_WORD = 3
CHUNK = 800
MW_MIN_ZIPF = 3.0       # each word of a French phrase must be this common
MW_MAX_PAIRS = 250      # per English word
EXTRA_DROP = {"-", "_", "ˈ", "ˌ", "‿", ".", "|", "‖", " ", "\t"}

# Native French words espeak wrongly pronounces as English. Correct IPA,
# hand-checked against standard French dictionaries.
FR_IPA_OVERRIDES = {
    "dos": "do", "vécut": "veky", "menthe": "mɑ̃t", "mythe": "mit",
    "mythes": "mit", "rythme": "ʁitm", "rythmes": "ʁitm",
    "rythmique": "ʁitmik", "ethnique": "ɛtnik", "ethniques": "ɛtnik",
    "ethnie": "ɛtni", "ethnies": "ɛtni", "labyrinthe": "labiʁɛ̃t",
    "zénith": "zenit", "méthane": "metan", "euthanasie": "øtanazi",
    "enthousiasme": "ɑ̃tuzjasm", "enthousiaste": "ɑ̃tuzjast",
    "enthousiastes": "ɑ̃tuzjast", "anthropologie": "ɑ̃tʁɔpɔlɔʒi",
    "anthropologue": "ɑ̃tʁɔpɔlɔɡ", "arithmétique": "aʁitmetik",
    "authentique": "otɑ̃tik", "authentiques": "otɑ̃tik",
    "authenticité": "otɑ̃tisite", "sympathisants": "sɛ̃patizɑ̃",
    "psychopathe": "psikɔpat", "bruns": "bʁœ̃", "emprunts": "ɑ̃pʁœ̃",
    "thune": "tyn", "tune": "tyn", "bled": "blɛd", "oued": "wɛd",
    "papy": "papi", "rallye": "ʁali", "chantilly": "ʃɑ̃tiji",
    "wallon": "walɔ̃", "wallonne": "walɔn", "wallons": "walɔ̃",
    "maghrébins": "maɡʁebɛ̃", "maghrébine": "maɡʁebin",
    "hyacinthe": "jasɛ̃t", "lithium": "litjɔm",
}
# Stray proper nouns that survived the dictionary filter.
FR_PROPER_DROP = {
    "henry", "jenny", "tommy", "williams", "stewart", "maxwell", "weber",
    "montmorency", "bush",
}
# Elision fragments that only exist with an apostrophe (qu', jusqu'...) —
# frequency lists strip the apostrophe and make them look like words.
FR_ELISION_FRAGMENTS = {
    "qu", "jusqu", "lorsqu", "puisqu", "quelqu", "presqu", "aujourd",
    "ll", "hui",
}


def clean(ipa: str) -> str:
    s = unicodedata.normalize("NFD", ipa)
    return "".join(ch for ch in s if ch not in EXTRA_DROP)


def load_dict_words(path: str) -> set[str]:
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip()
            if w and w == w.lower() and w.isalpha():
                words.add(w)
    return words


def batch_g2p(words: list[str], voice: str) -> tuple[dict[str, str], set[str]]:
    """Returns (word->IPA, words-that-language-switched)."""
    out: dict[str, str] = {}
    switched: set[str] = set()
    for i in range(0, len(words), CHUNK):
        chunk = words[i:i + CHUNK]
        text = "".join(w + ".\n" for w in chunk)
        r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--stdin"],
                           input=text, capture_output=True, text=True, check=True)
        lines = [ln for ln in r.stdout.split("\n") if ln.strip()]
        if len(lines) != len(chunk):
            lines = []
            for w in chunk:
                r1 = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, w],
                                    capture_output=True, text=True, check=True)
                lines.append(r1.stdout.strip())
        for w, ln in zip(chunk, lines):
            if "(" in ln:
                switched.add(w)
                ln = ln.replace("(en)", "").replace("(fr)", "")
            out[w] = clean(ln)
        print(f"  g2p {voice}: {min(i + CHUNK, len(words))}/{len(words)}",
              file=sys.stderr)
    return out, switched


def bigrams(ipa: str) -> frozenset:
    s = ("#",) + matcher._segs(_canonical(ipa)) + ("#",)
    return frozenset(s[i] + s[i + 1] for i in range(len(s) - 1))


@lru_cache(maxsize=None)
def feat_channel_cached(ipa_a: str, ipa_b: str) -> float:
    va = _variants(ipa_a)
    vb = _variants(ipa_b)
    return max(matcher._nw_sim(matcher._vecs(a), matcher._vecs(b)) for a in va for b in vb)


def combo(qi: str, ci: str) -> float:
    return 0.5 * _ngram_channel(qi, ci) + 0.5 * feat_channel_cached(qi, ci)


# -------------------------------------------------- alignment decomposition

def best_variant_pair(ipa_a: str, ipa_b: str) -> tuple[str, str]:
    best, pair = -1.0, (ipa_a, ipa_b)
    for a in _variants(ipa_a):
        for b in _variants(ipa_b):
            s = matcher._nw_sim(matcher._vecs(a), matcher._vecs(b))
            if s > best:
                best, pair = s, (a, b)
    return pair


def nw_traceback(ipa_a: str, ipa_b: str) -> list[tuple[str, str, float]]:
    """Aligned segment chunks (seg_a, seg_b, cost) on the best variant pair."""
    a, b = best_variant_pair(ipa_a, ipa_b)
    sa, sb = matcher._segs(a), matcher._segs(b)
    va, vb = matcher._vecs(a), matcher._vecs(b)
    n, m = len(va), len(vb)
    if n == 0 or m == 0:
        return []
    GAP = matcher.GAP
    sub = np.minimum(1.0, (np.abs(va[:, None, :] - vb[None, :, :]).sum(axis=2)
                           / (2.0 * matcher.N_FEATURES)) / matcher.SHARPEN)
    cost = np.zeros((n + 1, m + 1))
    cost[0, :] = np.arange(m + 1) * GAP
    cost[:, 0] = np.arange(n + 1) * GAP
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j - 1] + sub[i - 1, j - 1],
                             cost[i - 1, j] + GAP, cost[i, j - 1] + GAP)
    chunks = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(cost[i, j] - (cost[i - 1, j - 1] + sub[i - 1, j - 1])) < 1e-9:
            chunks.append((sa[i - 1], sb[j - 1], float(sub[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif i > 0 and abs(cost[i, j] - (cost[i - 1, j] + GAP)) < 1e-9:
            chunks.append((sa[i - 1], "·", GAP))
            i -= 1
        else:
            chunks.append(("·", sb[j - 1], GAP))
            j -= 1
    chunks.reverse()
    return chunks


def alignment_fields(ipa_a: str, ipa_b: str) -> dict:
    chunks = nw_traceback(ipa_a, ipa_b)
    if not chunks:
        return {"alignment": "", "shared_segments": 0, "coverage": 0.0}
    shared = sum(1 for x, y, c in chunks if x != "·" and y != "·" and c < 0.30)
    total = sum(1 for x, _y, _c in chunks if x != "·")
    return {
        "alignment": " ".join(f"{x}:{y}" for x, y, _ in chunks),
        "shared_segments": shared,
        "coverage": round(shared / max(1, total), 2),
    }


def main():
    t0 = time.time()
    n_en = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    n_fr = int(sys.argv[2]) if len(sys.argv) > 2 else 25000

    en_dict = load_dict_words("/usr/share/dict/american-english")
    fr_dict = load_dict_words("/usr/share/dict/french")
    en = [w for w in top_n_list("en", n_en)
          if w.isalpha() and len(w) > 1 and w in en_dict]
    fr = [w for w in top_n_list("fr", n_fr)
          if w.isalpha() and len(w) > 1 and w in fr_dict
          and w not in FR_PROPER_DROP and w not in FR_ELISION_FRAGMENTS]
    print(f"lexicons: {len(en)} EN x {len(fr)} FR", file=sys.stderr)

    en_espeak, _ = batch_g2p(en, "en-us")
    fr_espeak, fr_switched = batch_g2p(fr, "fr")
    for w, ipa in FR_IPA_OVERRIDES.items():
        if w in fr_espeak:
            fr_espeak[w] = clean(ipa)
            fr_switched.discard(w)
    loanwords = fr_switched  # remaining switched words are genuine anglicisms

    # Dictionary pronunciations first (Lexique 3 / WikiPron), espeak as a
    # second accent variant (EN: UK dictionary + US espeak are both real)
    # or as fallback for out-of-dictionary words.
    lex_en, lex_fr = load_lex_en(), load_lex_fr()
    en_prons: dict[str, list] = {}
    for w in en:
        ps = list(lex_en.get(w, []))
        e = en_espeak.get(w, "")
        if e and e not in ps:
            ps.append(e)
        en_prons[w] = ps[:2]
    fr_prons: dict[str, list] = {}
    for w in fr:
        ps = list(lex_fr.get(w, []))
        if not ps:
            e = fr_espeak.get(w, "")
            ps = [e] if e else []
        fr_prons[w] = ps[:2]
    en_ipa = {w: ps[0] for w, ps in en_prons.items() if ps}
    fr_ipa = {w: ps[0] for w, ps in fr_prons.items() if ps}

    # MUSE translations for cognate tagging
    translations: dict[str, set] = defaultdict(set)
    try:
        with open("/tmp/muse-en-fr.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    translations[parts[0]].add(parts[1])
    except FileNotFoundError:
        print("  (no MUSE dictionary; cognate tagging skipped)", file=sys.stderr)

    def is_cognate(e: str, f_: str) -> bool:
        tr = translations.get(e, set()) | translations.get(e.rstrip("s"), set())
        return f_ in tr or f_.rstrip("s") in tr

    fr_bg = [frozenset().union(*(bigrams(p) for p in fr_prons[w])) if fr_prons[w]
             else frozenset() for w in fr]
    index: dict[str, list[int]] = defaultdict(list)
    for j, bg in enumerate(fr_bg):
        for b in bg:
            index[b].append(j)
    fr_common = [(j, w) for j, w in enumerate(fr)
                 if zipf_frequency(w, "fr") >= MW_MIN_ZIPF and w not in loanwords]
    fr_seg_len = [len(matcher._segs(_canonical(fr_ipa[w]))) if w in fr_ipa else 0
                  for w in fr]

    def combo_multi(q_ps, c_ps):
        return max(combo(q, c) for q in q_ps for c in c_ps)

    entries = []
    for k, w in enumerate(en):
        q_ps = en_prons.get(w) or []
        if not q_ps:
            continue
        qi = q_ps[0]
        qb = frozenset().union(*(bigrams(p) for p in q_ps))
        if not qb:
            continue
        counts: Counter = Counter()
        for bgm in qb:
            counts.update(index[bgm])
        cands = sorted(((2 * c / (len(qb) + len(fr_bg[j])), j)
                        for j, c in counts.items()), reverse=True)[:SHORTLIST]
        scored = []
        for _dice, j in cands:
            cw = fr[j]
            if cw == w or not fr_prons.get(cw):
                continue
            s = combo_multi(q_ps, fr_prons[cw])
            if s >= B and counts[j] > 0:  # require >=1 shared exact bigram
                scored.append((s, cw, fr_ipa[cw]))
        scored.sort(reverse=True)

        best_single = scored[0][0] if scored else 0.0

        # ---- multi-word pass: only when no gold single match ----
        if best_single < S and len(matcher._segs(_canonical(qi))) >= 3:
            q_len = len(matcher._segs(_canonical(qi)))
            q_segs = matcher._segs(_canonical(qi))
            first_cands, last_cands = [], []
            for j, cw in fr_common:
                shared = counts.get(j, 0)
                if shared == 0 or fr_seg_len[j] >= q_len:
                    continue
                segs_c = matcher._segs(_canonical(fr_ipa[cw]))
                if not segs_c:
                    continue
                if segs_c[0] == q_segs[0]:
                    first_cands.append((shared, j, cw))
                if segs_c[-1] == q_segs[-1]:
                    last_cands.append((shared, j, cw))
            first_cands.sort(reverse=True)
            last_cands.sort(reverse=True)
            tried = 0
            best_mw = None
            for _s1, j1, w1 in first_cands[:20]:
                for _s2, j2, w2 in last_cands[:20]:
                    if w1 == w2:
                        continue
                    # exact segment-count match only: concatenation hiatus
                    # (wɛ+ɛn for "when") otherwise inflates loose phrases
                    if fr_seg_len[j1] + fr_seg_len[j2] != q_len:
                        continue
                    tried += 1
                    if tried > MW_MAX_PAIRS:
                        break
                    ci = fr_ipa[w1] + fr_ipa[w2]
                    s = combo(qi, ci)
                    # liaison variant: word-final s/x/z -> /z/, d/t -> /t/
                    segs2 = matcher._segs(_canonical(fr_ipa[w2]))
                    if segs2 and matcher._vecs(_canonical(fr_ipa[w2]))[0][0] == 1:
                        lc = "z" if w1[-1] in "sxz" else "t" if w1[-1] in "dt" else ""
                        if lc:
                            s2 = combo(qi, fr_ipa[w1] + lc + fr_ipa[w2])
                            if s2 > s:
                                s, ci = s2, fr_ipa[w1] + lc + fr_ipa[w2]
                    if s >= A and (best_mw is None or s > best_mw[0]):
                        boundary = fr_seg_len[j1]
                        best_mw = (s, f"{w1} {w2}", ci, boundary)
                if tried > MW_MAX_PAIRS:
                    break
            if best_mw is not None and best_mw[0] > best_single:
                s, phrase, ci, boundary = best_mw
                tier = "S" if s >= S else "A"
                e = {"en": w, "fr": phrase, "score": round(s, 3), "tier": tier,
                     "rank": 0, "multiword": True, "boundary_after_segment": boundary,
                     "cognate": False, "loanword": False,
                     "en_ipa": qi, "fr_ipa": ci, "en_freq_rank": k}
                e.update(alignment_fields(qi, ci))
                entries.append(e)

        for rank, (s, cw, ci) in enumerate(scored[:KEEP_PER_WORD]):
            tier = "S" if s >= S else "A" if s >= A else "B"
            if tier == "B" and rank > 0:
                continue  # B keeps best-per-word only
            loan = cw in loanwords
            if loan and tier in "SA":
                tier = "B"  # demote trivial anglicism matches
            e = {"en": w, "fr": cw, "score": round(s, 3), "tier": tier,
                 "rank": rank, "multiword": False, "boundary_after_segment": None,
                 "cognate": is_cognate(w, cw), "loanword": loan,
                 "en_ipa": qi, "fr_ipa": ci, "en_freq_rank": k}
            if tier in "SA":
                e.update(alignment_fields(qi, ci))
            entries.append(e)
        if (k + 1) % 1000 == 0:
            print(f"  ranked {k + 1}/{len(en)} ({time.time() - t0:.0f}s)",
                  file=sys.stderr)

    # ---- pairbank merge: score the user's historical 30k pair bank ----
    have = {(e["en"], e["fr"]) for e in entries}
    pb_added = pb_scored = 0
    try:
        import csv
        with open("data/pairbank.tsv", encoding="utf-8") as f:
            rdr = csv.DictReader(f, delimiter="\t")
            seen = set()
            for row in rdr:
                if row["tag"] != "homophone" or row["src_lang"] != "en":
                    continue
                pe, pf = row["src"].strip().lower(), row["tgt"].strip().lower()
                if pe == pf or len(pe) < 2 or len(pf) < 2:
                    continue  # identical loanwords / single letters: trivial
                if (pe, pf) in seen or (pe, pf) in have:
                    continue
                seen.add((pe, pf))
                if not pe.isalpha() or not pf.isalpha():
                    continue
                eps = lex_en.get(pe) or ([en_espeak[pe]] if pe in en_espeak else [])
                fps = lex_fr.get(pf) or []
                if not eps or not fps:
                    continue
                pb_scored += 1
                s = combo_multi(eps[:2], fps[:2])
                if s >= B:
                    tier = "S" if s >= S else "A" if s >= A else "B"
                    e = {"en": pe, "fr": pf, "score": round(s, 3), "tier": tier,
                         "rank": 0, "multiword": False, "boundary_after_segment": None,
                         "cognate": is_cognate(pe, pf), "loanword": False,
                         "pairbank": True,
                         "en_ipa": eps[0], "fr_ipa": fps[0], "en_freq_rank": 10**6}
                    if tier in "SA":
                        e.update(alignment_fields(eps[0], fps[0]))
                    entries.append(e)
                    pb_added += 1
    except FileNotFoundError:
        pass
    print(f"  pairbank: scored {pb_scored} unseen pairs, merged {pb_added} >= B",
          file=sys.stderr)

    entries.sort(key=lambda e: (-e["score"], e["en_freq_rank"]))
    counts_t = Counter(e["tier"] for e in entries)
    mw = [e for e in entries if e["multiword"]]
    cog = [e for e in entries if e["cognate"] and e["tier"] in "SA"]
    print(f"\n=== v3 dictionary: {len(entries)} entries from {len(en)} EN words ===")
    print(f"  S={counts_t['S']}  A={counts_t['A']}  B={counts_t['B']}")
    print(f"  multiword phrases: {len(mw)} (S/A only by construction)")
    print(f"  cognates (shared sound AND meaning, S/A): {len(cog)}")
    print(f"  loanword-demoted: {sum(1 for e in entries if e['loanword'])}")
    print(f"  took {time.time() - t0:.0f}s")

    with open("dictionary-v3.json", "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=0)
    with open("dictionary-v3.tsv", "w") as f:
        f.write("tier\tscore\ten\tfr\tflags\ten_ipa\tfr_ipa\talignment\n")
        for e in entries:
            flags = ",".join(x for x in [
                "multiword" if e["multiword"] else "",
                "cognate" if e["cognate"] else "",
                "loanword" if e["loanword"] else ""] if x)
            f.write(f"{e['tier']}\t{e['score']}\t{e['en']}\t{e['fr']}\t{flags}"
                    f"\t{e['en_ipa']}\t{e['fr_ipa']}\t{e.get('alignment', '')}\n")
    print("wrote dictionary-v3.json / dictionary-v3.tsv")


if __name__ == "__main__":
    main()
