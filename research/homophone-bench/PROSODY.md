# Prosody-aware sound judge + the linguistic tricks in play

`prosody.py` scores how well a French rendering sounds like the English source,
using phonetics the plain matcher ignored. The judge is now linguistically
informed and **diverged per language**.

## Implemented

1. **Stress weighting** — espeak's `ˈ`/`ˌ` marks (which the matcher stripped) set
   per-segment prominence: primary 1.0, secondary 0.6, unstressed 0.3. Mismatches
   on stressed segments cost full; unstressed 2nd/3rd syllables are cheaped out
   (`happy→happo` 0.43→0.75).
2. **Diverged EN/FR contour** — English is stress-timed and DECLINES (early/mid
   peak, low final); French is syllable-timed with PHRASE-FINAL prominence and an
   even rhythm. `english_naturalness` rewards an early peak + falling tail;
   `french_naturalness` rewards final-syllable prominence + evenness. A homophonic
   line is judged on the cross-lingual match × French-side naturalness, so
   "sounds right in English" and "sounds right in French" are scored separately
   (`un voile d'or`: english_contour 0.16, french_contour 0.85).
3. **Onset salience** — onset consonants inherit full syllable prominence, codas
   0.85× (onsets carry more perceptual identity).
4. **Rhythm / syllable-count match** — penalise differing syllable counts.
5. **Schwa/offglide leniency, rhotic map, nasal split** — already in `matcher.py`
   (English unstressed reduces to schwa; French r↔English ɹ; nasal vowel = V+n).
6. **Elision/liaison proposals** (LLM advisor, `ELISION_PROPOSAL.md`): schwa
   elision before vowel, enchaînement glides, geminate reduction, h-aspiré
   liaison block — pending a context-aware matcher to encode.

## Catalogue of further tricks (next)

- **F0 intonation contour** (actual pitch rise/fall), beyond lexical stress.
- **French final-consonant silence** and **liaison/enchaînement** across word
  boundaries (condition gaps on the boundary).
- **English flapping** (t/d→ɾ intervocalic; espeak already emits ɾ) and
  **aspiration**; **assimilation/coarticulation** at junctures.
- **Prosodic phrasing** — align where pauses/breaths fall so the line scans.

The reward used for self-learning is `prosody.prosodic_score × french_validity`
(`selflearn/reward.py`), so every trick above directly sharpens the teacher.
