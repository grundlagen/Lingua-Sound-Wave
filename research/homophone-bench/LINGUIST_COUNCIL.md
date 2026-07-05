# Linguistics council — meter & meaning improvement clues

_Saved advisories from multiple LLMs; review before encoding._


## DeepSeek

Here are concrete, codeable improvements for your homophonic-translation scorer, in two parts.

---

### (A) METER / PROSODY SCORING

1. **Syllable-weight penalty for French e-muet**  
   *Rule:* In French verse, a word-final schwa (ə) is pronounced only before a consonant or at caesura; otherwise it’s elided. Penalize if the engine places an e-muet in a position where it would be silent (e.g., before a vowel-initial word).  
   *Example:* EN `"boot"` → FR `"bout"` (no e-muet) is fine; but `"boute"` before a vowel would lose 0.3 points.  
   *Encoding:* After alignment, check each FR word-final `ə` against next word’s initial phoneme. Subtract weight if mismatch.

2. **Hemistich/caesura alignment**  
   *Rule:* French alexandrine has a mandatory pause after 6 syllables. English iambic pentameter has no fixed caesura. Reward if the FR output’s 6th syllable coincides with a major syntactic break (punctuation, conjunction).  
   *Example:* EN `"Humpty Dumpty sat on a wall"` → FR `"un petit un petit sur un mur"` (6th syllable = `"petit"` break) gets +0.2.  
   *Encoding:* Count syllables; if syllable 6 is followed by a comma/period/conjunction, add bonus.

3. **Rising vs. falling contour penalty**  
   *Rule:* English stress-timed peaks early/mid, then declines. French phrase-final prominence rises. Penalize if the FR output has a strong stress peak before the final syllable (i.e., mimics English contour).  
   *Example:* EN `"grasp"` (stress on first syllable) → FR `"grappes"` (stress on final /p/) is good; but if FR output places primary stress on syllable 1, subtract 0.4.  
   *Encoding:* Use espeak stress markers; if FR primary stress is not on the last syllable of the phrase, apply penalty.

4. **Secondary-stress handling**  
   *Rule:* English secondary stress (ˌ) often falls on 2nd/3rd syllables. French has no secondary stress. Penalize if the engine maps an English secondary-stressed syllable to a French syllable that would be fully unstressed (e.g., schwa).  
   *Example:* EN `"ˌɛk.səˈleɪ.ʃən"` → FR `"ɛk.sa.la.sjɔ̃"` – the ˌɛk maps to /ɛk/ (good), but if it mapped to /ə/ (schwa), subtract 0.3.  
   *Encoding:* For each English secondary-stressed vowel, check if French counterpart is a full vowel (not ə). If not, penalty.

---

### (B) MEANING SCORING

1. **Collocation coherence (bigram PMI)**  
   *Rule:* Score each adjacent FR word pair by pointwise mutual information (PMI) from a French corpus. Low PMI (e.g., `"un petit"` is fine; `"petit mur"` is fine; `"mur un"` is nonsense) penalizes.  
   *Example:* EN `"boot"` → FR `"bout"` (alone) is fine; but if engine outputs `"bout de"` (PMI high) vs `"bout zèbre"` (PMI near zero), subtract 0.5.  
   *Encoding:* Precompute PMI table from French Wikipedia; sum log-PMI over bigrams; normalize by length.

2. **Semantic coherence (word2vec cosine)**  
   *Rule:* For each FR word, compute cosine similarity to the previous word’s embedding. If the sequence is semantically random (e.g., `"bout mur"`), penalize.  
   *Example:* EN `"grasp"` → FR `"grappes"` (fruit) → next word `"de"` (of) is fine; but `"grappes mur"` (wall) is weird → -0.4.  
   *Encoding:* Use fastText French embeddings; average cosine over all adjacent pairs; threshold <0.2 → penalty.

3. **Image/poetic plausibility (noun-verb ratio)**  
   *Rule:* Poetic French tends to have a higher noun-to-verb ratio than prose. If the output is all verbs or all prepositions, it’s likely nonsense. Reward if noun count > verb count.  
   *Example:* EN `"Humpty Dumpty"` → FR `"un petit un petit"` (nouns) gets +0.3; if it became `"un petit tombe"` (verb), no bonus.  
   *Encoding:* Tag POS with spaCy; compute ratio; if ratio >1.5, add bonus.

4. **Gap penalty for function-word overload**  
   *Rule:* If the output contains more than 40% function words (articles, prepositions, pronouns), it’s likely meaningless filler. Penalize.  
   *Example:* EN `"the boot"` → FR `"le bout"` (50% function) is borderline; but `"le de la"` (100%) → -0.6.  
   *Encoding:* Count function words from a stoplist; if proportion >0.4, subtract (proportion-0.4)*2.


## Gemini

(unavailable)


## Nemotron

**A. METER / PROSODY SCORING (French‑vs‑English)**  

| # | Rule (what to measure) | IPA / word example showing the contrast | How to encode it (weight / gap / feature / model) |
|---|------------------------|------------------------------------------|---------------------------------------------------|
| A1 | **Syllable‑weight penalty** – French treats most syllables as light; English heavy (coda‑consonant or diphthong) syllables should be down‑weighted if they appear in a French‑like position. | EN *ˈbʊt* → FR *bu* (light) vs EN *ˈbʊt* kept as *buːt* (heavy). | Compute a weight *w* = 1 – (0.3 × #heavy‑in‑FR‑slot). Add *w* to the prosody score; heavy syllables in strong French beats get a larger penalty. |
| A2 | **E‑muet (silent e) count** – In verse, each optional schwa that can be realized adds metrical flexibility; reward outputs that place a schwa where French poetry allows elision. | EN *ˈæpəl* → FR *ə.pəl* (schwa before *p*). | Scan the French phoneme string for schwa that follows a consonant and precedes a vowel; give +0.2 per eligible schwa (capped at +0.6 per line). |
| A3 | **Hemistich / caesura alignment** – French alexandrine expects a pause after the 6th syllable; English stress‑timed peaks often fall elsewhere. Measure distance of the nearest primary stress to the canonical caesura. | EN *ˈhʌm.pi ˈdʌm.pi* → FR *œ̃.pə.ti œ̃.pə.ti* (stress on syllable 3 & 6). | Compute *d* = |stress‑syllable – 6|; score contribution = exp(–*d*/2). |
| A4 | **Rising vs falling contour** – French phrase‑final prominence is a slight rise; English declination is a fall. Reward a final‑syllable F0 slope that matches the target language. | EN *ˈɡɹæs* (fall) → FR *ɡʁas* (slight rise). | Use a lightweight pitch‑contour estimator (e.g., Praat‑like slope over last 50 ms); add +0.3 if slope > 0 (rise) for French, –0.3 if slope < 0 (fall) for English. |
| A5 | **Secondary‑stress smoothing** – English secondary stress (ˌ) often maps to a weak French beat; penalize secondary stresses that land on strong French positions (1, 3, 5, 7 in an alexandrine). | EN *ˌɛkˈsplɔʒən* → FR *ɛk.splɔ.ʒɑ̃* (secondary on syllable 1). | For each secondary stress, if its syllable index mod 2 = 1 (odd) in French meter, subtract 0.15; otherwise add 0.05. |
| A6 | **Syllable‑count match with tolerance** – Allow ±1 syllable to accommodate elision or liaison, but apply a linear penalty beyond that. | EN *ˈbʊtˌɡɹæs* (3 syll) → FR *bu ɡʁas* (2 syll). | Score = 1 – 0.2 × |Δsyll| if |Δsyll| ≤ 2, else 0. |

---

**B. MEANING SCORING (beyond surface validity)**  

| # | Rule (what to measure) | IPA / word example showing the contrast | How to encode it (weight / gap / feature / model) |
|---|------------------------|------------------------------------------|---------------------------------------------------|
| B1 | **Collocation likelihood** – Use a French bigram‑log‑probability from a large corpus (e.g., frWaC). Reward high‑probability adjacent pairs. | EN *ˈɡɹæs ˈhɒp* → FR *ɡʁas o.p* (grass‑hop) → bigram “herbe saut” has low prob; better “saut d’herbe”. | Compute *logP* = log P(w₂|w₁); add *α* × logP (α≈0.4). |
| B2 | **Semantic coherence via embeddings** – Average cosine similarity of content‑word embeddings (French fastText) to a topic vector derived from the source English line. | EN *ˈhʌm.pi ˈdʌm.pi* → FR *œ̃.pə.ti œ̃.pə.ti* (nonsense) → low similarity to “enfant” theme. | Topic vector = mean of embeddings of content words in EN line; score = β × mean cosine (β≈0.5). |
| B3 | **Image‑concreteness bonus** – Prefer outputs that contain at least one concrete noun (image‑evoking) from a French concreteness lexicon. | EN *ˈbʊt* → FR *bu* (no concrete noun) → add *boot* → *botte* (boot) gives image. | If any token’s concreteness > 0.6 (norm 0‑1), add +0.2; else 0. |
| B4 | **Poetic device detection** – Reward internal rhyme, alliteration, or assonance that mirrors the source’s sound play. | EN *ˈstʌm.pɪd ˈɡɹæs* → FR *ɛ̃.tɑ̃ pɥi ɡʁas* (internal nasal ɛ̃). | Scan for repeated phoneme patterns within a 3‑syllable window; add +0.15 per detected device (capped 
