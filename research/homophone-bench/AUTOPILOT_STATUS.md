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
