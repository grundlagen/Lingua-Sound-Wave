# The 1+2+3 generation engine: resegmentation under coherence

`generation_engine.py` ties together resegmentation + coherence + content
selection, building ONLY on what June 11–12 perfected. It is the operation a
word-for-word table cannot do (re-cut the phoneme stream); the dictionary is the
parts bin, this is the assembler.

## What June 11–12 perfected (reused verbatim, not reinvented)

**Resegmentation primitives — `phonetic_decoder.py` (v4, Jun 11).** The beam
decoder walks an English phoneme stream and emits *French* words. The key line:
word boundaries cost **nothing acoustically** ("space understanding"; only a
small `WORD_PENALTY` 0.18 + `FREQ_BONUS`), so French boundaries fall where French
words land — not where the English words were. Plus `MIN_WORD_SEGS` (no confetti)
and `LIAISON` z/t/n bridges. This *is* the oronym/juncture engine.

**Phonetics rules — `matcher.py` (Jun 10–11).** Carried into every decode:
`EQUIV` floors (voicing p/b·t/d·k/ɡ at 0.20; lax i/ɪ·e/ɛ at 0.10; TH→s/f/t;
ŋ→n; front-rounded y/ø; glides), the `SHARPEN` ÷0.35 trick, `RHOTIC_MAP`
(ʁ ʀ ɾ r→ɹ), `NASAL_SPLIT` (ɑ̃→ɑn), `DIPHTHONG_SMOOTH` (eɪ→e), `CHEAP_GAP`
(offglide/schwa/h), and the stop place-of-articulation floor.

**Junction rules — `generate.py` / `finalize.py` (Jun 12).** `hiatus` (no FR
coda-V meeting FR onset-V across a join), `fr_onset`/`fr_coda` classes,
`syllable_delta`. Applied when chaining kept spans.

## What the engine adds (only the glue)

1. **(1) RESEGMENT** — reused `pd.decode` re-cuts each span (free boundaries).
2. **(2) COHERENCE** — reused bigram-LM *in the beam* + as the re-rank, so the
   re-cut lands on fluent French. (Still the weak link; the interface is the
   point — drop in a real L2 LM/LLM and nothing else changes.)
3. **(3) CONTENT SELECTION** — slide phrase windows, resegment each, score
   `dual = sound × coherence`, **keep** spans that carve, **flag** the rest as
   synonym-swap candidates (the future layer; exposed as a hook, not built).

## Tested behaviour (public-domain 1916 source)

Resegmentation demonstrably moves boundaries:

| EN span | re-cut FR | boundary change | dual |
|---|---|---|---|
| "up the hill" | as il | 3 EN → 2 FR | 0.65 |
| "Horner sat" | corner | 2 EN → 1 FR | 0.50 |
| "in a" | les | 2 EN → 1 FR | 0.64 |

Content selection keeps those and flags the rest: `Humpty`, `Dumpty`, `Jill`,
`Little`, `wall` come back as **GAP → synonym-swap candidates** — exactly the
non-lexical rhyme words with no clean carve. So the engine produces *fragments*,
not verse yet, for two known reasons, both already located:

- **coherence is weak** (bigram) — the gating component (impetus I);
- **non-lexical words don't re-cut** — which is what the **synonym-swap layer** is
  for.

## Where the future synonym-swap plugs in (the owner's vision)

The hook is already the GAP/weak verdict. The future layer: understand the
rhyme's overall meaning, then for each GAP/weak span substitute a **near-synonym**
(or re-phrasing) whose phoneme string *does* align on both sides — "phoneme stream
found matching on both sides by whatever method." Meaning is preserved at the
rhyme level; the sound is made to carve. That is `paraphrase_translate.py`'s
move (open the source side) promoted into the per-span engine, and it sits
cleanly on top of 1+2+3 without changing them.

## Honest status

The engine is the right *operation* and reuses every perfected piece. It works:
boundaries re-cut, the LM steers, selection separates carve-able spans from
synonym candidates. It does not yet make whole rhymes, because the coherence
model is weak and the synonym layer isn't built — the two items 1+2+3 explicitly
scopes next. No dictionary change can close that gap; this engine is where the
work now lives.
