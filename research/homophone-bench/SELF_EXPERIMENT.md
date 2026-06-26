# Self-experiment & idea trawl — preserving old code

The engine should improve itself without ever damaging what works. Three tools,
all **read-only on source**:

- **`idea_trawler.py`** — scans every module's docstring across the current tree
  AND every git branch (and notebook markdown), plus a Nemotron synthesis of the
  most underused ideas → `IDEAS.md`. Found 22 ideas living only on other branches
  (e.g. `multilang.py` one-config button, `periphrastic_translate.py`,
  `fragment_route.py` IPA-chunk tunnelling, `llm_layer.py` DeepSeek fluency).
- **`experiment.py`** — sweeps parameters of the reward/prosody/matcher by
  monkeypatching in-memory constants at RUNTIME (never editing .py) and logs
  good-vs-bad separation to `experiments/results.jsonl`. First run: prosody is
  more lenient than plain combo on a tiny set (a real signal to harden the
  discrimination, surfaced without changing anything).
- **`linguist_council.py`** — multi-LLM advisories (DeepSeek/Gemini/Nemotron),
  saved to `LINGUIST_COUNCIL.md` for review.

## The preservation protocol

1. **Never overwrite.** Experiments live in `experiments/` and as runtime
   parameter sweeps. Ideas ported from other branches arrive as NEW files
   (`module.py` stays; a variant is `module_v2.py`), never edits-in-place.
2. **Branch off.** Run exploratory work on `selflearn-experiments`; main code
   stays on `claude/phrase-weave-multiword`. Old code is always recoverable from
   git history and from the other branches the trawler indexes.
3. **Promote deliberately.** A sweep/council finding is applied only by a
   reviewed edit, with the previous version preserved (git + a kept copy).

Self-improvement = trawl → council → experiment (logged) → review → deliberate
promote. The training loop (`run_continual.py`) bootstraps the model; these tools
bootstrap the *engine*, safely.
