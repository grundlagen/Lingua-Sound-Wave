# Lingua-Sound-Wave — Claude session brief

You are working on **dual translation**: text that, read aloud, SOUNDS like
English while reading as coherent French — ideally MEANING it too (literal ∧
homophonic at once). Word scope is solved (9,231 DUAL-S pairs); sentence scope
runs ~50–66% joint; the goal is to raise it. Rupert's project; paper-mode
research, public-domain sources only.

## The one law
**Proposers propose; the judge disposes.** No LLM (including you) ever scores
sound. Sound is scored by `matcher.py` combo (judge of record, AUC 0.993),
meaning by `semantic_cosine.py` (MiniLM), French-ness by the Lexique gate.
Every generated line must be machine-verified before it is claimed. Beware the
two measured artifacts: franglais leak (English words score high on both
channels — always Lexique-gate) and MUSE FR-column noise.

## Read these first (all in `research/homophone-bench/`)
1. `read me first` (repo root of the bench) — the 6-stage pipeline map
2. `METHODS_STATUS.md` — what is LIVE / READY / DORMANT / SUPERSEDED
3. `RUN_ON_GPU.md` — your run plan if this box has a GPU
4. `MATHEMATICS.md` — the formal foundations + ranked transfer queue
5. `IDEAS_BABEL.md` — the 47-route catalog with honest bench ledger
6. `DUAL_SCALE.md` — the scope law + the paraphrase-inversion result

## Setup
```bash
cd research/homophone-bench
bash setup_gpu.sh          # deps, data fetches, corpus build, smoke test
```
Secrets: `.env.local` (gitignored) holds API keys — load with
`import _load_env; _load_env.load_keys()`. NEVER commit it. If keys are
missing, symbolic pipeline still runs fully offline.

## Your in-the-loop roles (you replace Haiku where you're stronger)
- paraphrase proposer (10 varied FR + 6 EN rewordings; see
  `paraphrase_search.py` prompts)
- sound-bender (rewrite FR toward a target phoneme stream; verify with combo)
- grammar fixer (sound-preserving repairs only; see `dual_poet.haiku_fix`)
- arbitration on `strict_judge.py` disagreement cases
Always verify your own proposals with the judge before keeping them.

## Priority queue (top first)
1. MERT/logistic per-channel calibration on strict-gold (closes composer
   50%→55%+; `MATHEMATICS.md` §6 — it's one logistic regression)
2. GPU: train the dual model — `python build_train_corpus.py` then
   `selflearn/train_selflearn.py --data ../train-dual-v1.jsonl` (168k rows)
3. GPU: TRUE constrained decoding (automaton intersection via logit-bias;
   port `constrained_poet.py`; design in `METHODS_DEEP_DIVE.md`)
4. WMD/Sinkhorn paragraph meaning judge; fuse `set_dual.py` (submodular
   cover) with `paraphrase_search.py` (meaning-first seeds)
5. Knuth–Bendix canonical forms for the connected-speech rules (~100× match
   speedup; also fixes the window-index bench timeout)

## Working rules
- Branch: work on the branch you were given; commit+push often; never commit
  `.env.local`, `.gcp-sa.json`, `*.pkl` LMs, `train-dual-v1.jsonl` (all
  gitignored; regenerate with the scripts).
- Record every bench honestly in the relevant ledger (IDEAS_BABEL /
  DUAL_SCALE) — including regressions. The ledgers only work if they're true.
- Long jobs: run in background, poll the output file, keep working.
