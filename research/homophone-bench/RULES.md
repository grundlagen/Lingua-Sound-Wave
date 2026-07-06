# MANDATORY ARCHITECTURE — DO NOT DEVIATE
# Extracted from user's messages across this session.
# Commit: if you're reading this, these rules are NON-NEGOTIABLE.

## RULE 0: THIS IS NOT TRANSLATION
Homophones = words that SOUND alike (measured by combo/ndice).
Translations = words that MEAN the same (measured by semantic cosine).
NEVER mix them. Never put MUSE entries in sound edges.
Never map "and→et" — that's a translation.
Map "and→inde" (0.60 sound) — that's a homophone from zipf-glue.tsv.

## RULE 1: FOUR SEQUENTIAL STAGES
Each stage produces a MASSIVE database. Next stage feeds on it.
  Stage 1 → stage1_homophones.jsonl (word pairs, sound-filtered)
  Stage 2 → stage2_graph.pkl (meaning web, relationships)
  Stage 3 → stage3_*.jsonl (periphrastic phrases, sentences)
  Stage 4 → paragraphs (dual-language coherence)
NOT a single pipeline. NOT HMFBO. Sequential stages.

## RULE 2: ENGLISH CAN CHANGE + FRENCH CAN CHANGE
Recursive algebraic graph. Both sides shift.
"the sea remembers every ship" → English can become "ocean recalls each vessel"
→ French adapts → English adapts again. Reciprocal.

## RULE 3: PERIPHRASTIC = PARAPHRASING WITH AVAILABLE WORDS
Not word-for-word. Use MORE words to express meaning indirectly.
Use fragments (84K fr-units), babel windows (multi-word merges),
meter, poetry mode, filler words. Generate endless variations.
The 347K overnight phrases are word-for-word — that's WRONG.

## RULE 4: MULTI-WORD TO MULTI-WORD
2-3 English words → 1 French word (babel many→one)
1 long English word → 2 French words (babel one→many)
Use whole_line_carve.py for line-level beam search.

## RULE 5: BOTH SIDES MUST HAVE MEANING
Every French output must pass: real French vocabulary gate +
French bigram LM fluency + FR→EN meaning backlink.
Every English output must pass: real English vocabulary gate +
English bigram LM fluency + EN→FR homophone quality.

## RULE 6: DON'T ACCEPT BAD RESULTS AS GOOD
"beat beet bead" → "bittent bittes bides" is a CLUSTER TRAP.
Words all from the same sound family = no semantic diversity.
If sound > 0.9 but meaning diversity < 2 unique families, REJECT.

## RULE 7: USE WHAT EXISTS ON GITHUB
phrase_bank.py (3K carved phrases), compose_lots.py (DP fragment assembly),
whole_line_carve.py (line-level beam search), poetry_mode.py (meter + fillers),
phonetic_decoder.py (52K-word trie), babel_windows.py (window merges),
fragment_weave.py (shared IPA blocks), wire_everything.py (periphrastic edges),
matcher.py (AUC 0.993 combo scorer), bigram_lm.py (fluency scoring).

## RULE 8: GPU TRAINING MUST PERSIST
auto_save.sh pulls checkpoint → local → GitHub every 5 min.
Checkpoint at every epoch (train_transformer_v2.py).
If GPU dies, restart from last checkpoint.

## RULE 9: DON'T MAKE ME REPEAT MYSELF
Every time one of these rules is violated, I have to say it again.
Read this file before making any architectural decision.
