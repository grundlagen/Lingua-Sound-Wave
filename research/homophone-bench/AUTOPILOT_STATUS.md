## supervision @ 2026-07-03 00:40 UTC (web session)
- branch synced at 4fa6eb3; no new Colab/terminal commits since 9073ca1 (T4 memory fit).
- NO heartbeat anywhere: no TRAIN_ERRORS.log, no RESULTS.tsv, no selflearn-status branch.
- Likely cause: run_continual's status push needs GITHUB_TOKEN + GITHUB_REPO env vars
  set in the Colab cell — without them gh_put() silently no-ops, so a healthy OR dead
  loop both look dark. ACTION for Rupert: add to the Colab cell before launch:
      import os; os.environ["GITHUB_REPO"]="grundlagen/Lingua-Sound-Wave"
      os.environ["GITHUB_TOKEN"]="<a fine-grained push token>"
  (or just re-run the one-cell colab_selflearn.ipynb, which pushes via git directly).
- If the trainer restarted after 4fa6eb3 it now trains on the 168k dual corpus with
  the eval harness; first RESULTS.tsv row is the confirmation to look for.

## supervision @ 2026-07-03 ~11:00 UTC — CHECKPOINT FOUND (greerhalfenstadt Drive)
- homophonic-carver/: real Qwen2.5-1.5B checkpoint (3.09GB safetensors + config +
  tokenizer), warm-started once at 2026-07-02 19:59.
- status.json: round 894, error "" at 00:29 — BUT model.safetensors not re-saved
  since 19:59. Diagnosis: expert-iteration STALL — 894 rounds kept nothing above
  keep_thresh=0.55, so no SFT, model frozen at warm-start. ~18s/round = no-op spin.
- lingua-ckpt/: empty (today's run hasn't saved a warm-start yet).
- No RESULTS.tsv/meta.json => checkpoint predates the eval harness; never measured.
FIXES (this commit): warm-start + resume now run eval_harness immediately (first
RESULTS.tsv row); adaptive keep_thresh (starts 0.42, anneals to 0.35 floor when a
round keeps nothing, ratchets up to the target as the model improves) — unblocks
the cold-start stall. Next Colab run on the dual corpus will warm-start into
lingua-ckpt, self-measure, and actually learn past round 0.

## session @ 2026-07-12 (web, branch claude/homophone-writer-upgrade-sj1cx0)
- previous branch merged+deleted; branch restarted from main per workflow.
- env: CPU-only box; judge smoke OK (combo=1.000); fixed 40 hardcoded
  /home/mint paths; restored adversarial strict_judge.py (merge had clobbered
  it with the gate script, now strict_gate.py).
- queue #1 DONE: calibrate_channels.py reruns clean — logistic AUC 0.795 vs
  geo 0.786, strict gold-rate 53.3% vs 52.7%; channel-calibration.json fresh.
- stage 2 DONE: build_ladder_json.py -> ladder-words.jsonl (93k words).
- stages 3+4 deterministic DONE: inflect_expand.py -> inflection-pairs.tsv,
  +4,811 DUAL-S (3,792 non-cognate) / +20,211 DUAL-A; corpus now 179k rows.
- NEW: fable_writer.py — judge-verified fable-scale writer (anthropic/ollama/
  in-the-loop backends); demo bench 3/6 lines VERIFIED (ledger §H).
- next for GPU box: retrain on enlarged corpus; regen node-vecs for senses[].
