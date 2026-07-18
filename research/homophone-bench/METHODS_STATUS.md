# Methods audit — what runs, what waits, what was superseded

_Every method in the repo, honestly triaged. LIVE = wired into the current
pipeline; READY = works, run on demand; DORMANT = built, superseded or waiting
on env; DEAD = measured worse, kept for the record._

## LIVE (fire on every compose)
| method | file |
|---|---|
| combo matcher (judge of record) | `matcher.py` |
| rule-aware realizations (flap/th/h/yod/schwa/apocope/cluster) | `rule_aware.py` |
| juncture (liaison/elision cross-word, verify credit) | `juncture.py` |
| greedy channel composer (dual/ladder/glue/chains/haiku/classes/windows) | `beauty_compose.py` |
| real-cosine calibration of top-K | in `beauty_compose.candidates` |
| homophone classes ×2 languages + Lexique-authoritative | `*-homophone-classes*.tsv` |
| window merges (many→one, units incl. elision/liaison/compounds) | `babel_windows.py` |
| trigram L2 (bake-off winner) | `trigram_lm.py` |
| meaning channel (MiniLM) + Lexique gate | `semantic_cosine.py` |
| meaning-first paraphrase search + coordinate ascent | `paraphrase_search.py` |
| verify-constrained decoding (E40 env-honest) | `constrained_poet.py` |
| Haiku bridge/kenning/metonym/antonym miner (judge-verified) | `llm_bridge.py` |

## READY (run on demand, current)
sentence former — POS grammar + agreement/elision/fusion repairs + juncture
rescore (`sentence_former.py`, Lexique 3.83 via pylexique; `SENTENCE_FORMER.md`) ·
sentence self-improve loop over corpus-phrases DB (`sentence_selfimprove.py`) ·
strict/hard judging (`strict_judge.py`, `hard_judge.py`) · self-improvement
loop (`self_improve.py`) · tier ladder build (`tier_ladder.py`) · dual miners
EN→FR/FR→EN/ES (`dual_mine.py`, `babel_es.py`) · training corpus
(`build_train_corpus.py` → 168k rows) · rhyme families (`rhyme_pick.py`) ·
real-audio G2P validation (`real_audio_g2p.py`) · CPU neural carver
(`selflearn/neural_carver.py`) · zipf glue miner (`zipf_glue.py`).

## DORMANT (waiting on GPU/box — see RUN_ON_GPU.md)
GPU SFT + expert iteration (`selflearn/train_selflearn.py`, `run_continual.py`,
colab notebook) · TRUE constrained decoding with logit-bias (E40 full form) ·
WordNet hierarchies (C23, proxy-blocked) · Common-Voice-scale audio mining.

## SUPERSEDED (kept, understood)
round_rabbit lattice (→ dual-pairs + chains) · Viterbi trigram composer
(benched 23% vs greedy 55%) · mfcc/audio retrieval (AUDIO_INVESTIGATION: audio
re-ranks, doesn't retrieve) · bigram LM (→ trigram) · embeds as fluency
(coin-flip on word order — meaning channel only) · v5 rerank / v6 build
(→ v7 → tier-ladder).

## NOT forgotten but genuinely open
per-channel logistic calibration (top of queue) · paragraph-level set cover
(NEW — `set_dual.py`) · rhyme-scheme composition using rhyme_pick · polysemy
split C28 · assonance-in-beam E37 (metric exists) · PanLex/Wiktionary F44 ·
window-index speed tier (82k units outgrew the bench timeout).
