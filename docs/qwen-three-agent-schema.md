# Qwen three-agent homophone loop — training & interaction schema

How to turn the three-agent homophone system (A: EN→FR carver, B: FR→EN
"English ear", C: judge) into fine-tuned Qwen agents, using the data the
pipeline already produces (`dictionary-v7-integrated.json`, the phrase bank,
`corpus-carves.tsv`, and the AUC-0.993 combo scorer).

The one-line design: **A and B are small fine-tuned Qwen models; C is NOT an
LLM** — C is the combo scorer + embeddings you already trust, emitting
structured feedback that A consumes for span-level repair. The combo scorer
doubles as a verifier for preference training, which is where the real
quality gain over plain SFT comes from.

---

## 1. Roles and the loop protocol

```
        EN text ─────────────► AGENT A (Qwen, EN→FR carver)
                                    │  French line
                                    ▼
                               AGENT B (Qwen, FR→EN hearer)
                                    │  "heard" English
                                    ▼
                               AGENT C (deterministic judge)
                                    │  JSON verdict + misaligned spans
                                    ▼
                    accept  ◄── loop back to A with span repairs (max 3 rounds)
```

### The Agent C fix (why suggestions weren't firing)

C must run **two separate comparisons** — the earlier bug was collapsing them
into one against the original English:

1. **Sound loop fidelity** (the homophone test): align B's heard English
   against the **original** English, phoneme-level, using combo. This is the
   score that decides accept/reject.
2. **Repair targeting** (the suggestions): diff B's heard English against the
   **current drifted English** — i.e. whatever English reading A's *latest*
   French actually produces — not the round-0 original. Suggestions fire on
   spans where the current French's sound has drifted, which after round 1 is
   a different set of spans than the original diff. Comparing against the
   original forever means round-2+ suggestions point at spans A already fixed,
   so nothing actionable fires.

Concretely: keep `en_original` (fixed) and `en_heard_prev` (B's output from
the previous round). Score against `en_original`; target repairs by aligning
`en_heard_current` vs `en_heard_prev` + `en_original` jointly, and only emit a
suggestion for a span whose combo against `en_original` is below floor AND
changed (or failed to improve) since the previous round.

### C's verdict schema (what A receives back)

```json
{
  "sound_combo": 0.61,
  "meaning_cos": 0.18,
  "verdict": "revise",
  "spans": [
    {
      "en_span": "and mouse",
      "en_ipa": "ə n d   m aʊ s",
      "fr_span": "vaisselle date",
      "heard": "sell date",
      "span_combo": 0.22,
      "candidates": ["en mousse", "an moussent"]
    }
  ]
}
```

`candidates` come from your existing carve pool (`poetry_mode.py` trie /
phrase bank lookup on the span's phonemes) — C hands A a shortlist, A picks
the one that keeps the French rail fluent. That keeps the LLM doing what LLMs
are good at (fluency arbitration) and the scorer doing what it's good at
(sound truth).

---

## 2. Agent A — EN→FR carver (Qwen SFT)

**Model**: Qwen3-4B-Instruct (or 1.7B if latency matters; below that,
phoneme fidelity degrades). LoRA r=16, α=32, lr 1e-4, 2–3 epochs.

**Training data**: every verified carve you already have — phrase bank
(`phrase-bank-balanced.tsv`), corpus carves, dictionary v7 pairs — filtered to
combo ≥ 0.45. Include the IPA in the prompt: giving the model the phoneme
stream explicitly is the single biggest lever, because the task is phoneme
carving, not translation, and Qwen's tokenizer hides sound structure.

**JSONL (ChatML via `messages`)**:

```json
{"messages": [
  {"role": "system", "content": "You carve English sound into French. Given English text and its IPA, output French words that, read aloud in French, sound like the English. The French must be real French words. Output only the French."},
  {"role": "user", "content": "EN: humpty dumpty sat on a wall\nIPA: h ʌ m p t i  d ʌ m p t i  s æ t  ɒ n  ə  w ɔː l"},
  {"role": "assistant", "content": "un petit d'un petit s'étonne aux Halles"}
]}
```

**Repair turns** (round 2+ of the loop) are trained as multi-turn
continuations of the same conversation — this is what makes A actually usable
inside the loop rather than only zero-shot:

```json
{"messages": [
  {"role": "system", "content": "…same system prompt…"},
  {"role": "user", "content": "EN: cat and mouse\nIPA: …"},
  {"role": "assistant", "content": "quatre vaisselle date"},
  {"role": "user", "content": "REVISE. Keep: \"quatre\". Fix span \"and mouse\" (heard as \"sell date\", combo 0.22). Candidates: en mousse | an moussent"},
  {"role": "assistant", "content": "quatre en mousse"}
]}
```

Generate these repair examples synthetically: take a good carve, corrupt one
span with a low-combo alternative from the decoder's n-best list, and the
"fix" is the original. Cheap, unlimited, and exactly the distribution the
loop produces.

## 3. Agent B — FR→EN hearer (Qwen SFT)

Same recipe, reversed — but do **not** just flip A's pairs. B's job is "what
does this French *sound like* to an English ear", which is a phonetic
rendering task, so its gold outputs should be built by running the French
through FR G2P and then your EN carve machinery in reverse (segment the FR
phoneme stream into EN words with the same beam decoder, EN trie instead of
FR). Train B on those, plus the human-verified pairs reversed.

```json
{"messages": [
  {"role": "system", "content": "You are an English ear. You hear French text as if it were English. Output the English words the French sounds like when read aloud in French. Do not translate."},
  {"role": "user", "content": "FR: quatre en mousse"},
  {"role": "assistant", "content": "cat on moose"}
]}
```

Critical: keep B **honest, not helpful**. If B is trained only on
gold homophone pairs it learns to "solve the puzzle" and reports the intended
English even when the French drifted — which silently breaks C. Mix in ~30%
non-homophonic ordinary French with faithful phonetic renderings as targets,
so B reports what it hears, not what A hoped.

## 4. Making it *better* than SFT: the verifier loop

This is the actual answer to "best schema for a better agent from Qwen":
you own a trusted verifier (combo, AUC 0.993), so use **rejection-sampling +
DPO** on top of SFT — no reward model needed:

1. Sample A at temperature 0.9, n=16 per English input.
2. Score every sample: `reward = combo(EN, B(fr)) + λ·french_fluency(fr)`
   (bigram LM or a French LM for fluency; λ ≈ 0.3).
3. Best sample → new SFT example (rejection sampling / best-of-n distill).
4. (best, worst) sample pairs with reward gap ≥ 0.15 → DPO pairs:

```json
{"prompt": [{"role":"system","content":"…"},{"role":"user","content":"EN: …\nIPA: …"}],
 "chosen":   [{"role":"assistant","content":"un petit d'un petit"}],
 "rejected": [{"role":"assistant","content":"une pitié dompter"}]}
```

5. One DPO epoch (β=0.1) on the SFT checkpoint; iterate 2–3 rounds.

Same loop applies to B, with reward = phonetic faithfulness of B's output
against the FR phoneme stream (not against the intended English — see
"honest, not helpful").

## 5. What NOT to make an LLM

- **Agent C**: deterministic (combo + embedding cosine + span aligner). An
  LLM judge here would be strictly worse than the scorer you benchmarked, and
  it's the component whose honesty the whole loop depends on.
- **G2P**: keep `lexicon_g2p.py` / dictionary lookup. Feed its output into
  the prompts; don't ask Qwen to imagine IPA.

## 6. Loop budget & stopping

- Max 3 A↔B↔C rounds per line; accept at `sound_combo ≥ 0.55`, else keep the
  best round's output (never the last round's — track argmax).
- Every accepted line goes into the phrase bank / SFT pool → the flywheel:
  the loop generates its own next training set, verifier-filtered.
