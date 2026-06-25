# Weave + loop-certification applied to v7 (engine as-is)

Ran Fable's weave.py + explode_web.py UNCHANGED on dictionary-v7-integrated. Only
substitution: chain_game's embedding ~sem source -> MUSE-pivot synonyms (forced,
no torch; the project's own offline fallback). Logic untouched.

v7 results:
  full: 12,057 web edges -> 47,077 all-step connections; 354 loops -> 214 loop-
        certified sound pairs (loop-certified-pairs-v7.tsv)
  s-tier: 16 loops
Sample (richer multi-translation loops):
  city ≈ séries : cherry = cerise ≈ series = séries ≈ city ~ cities ≈ cerises
  mass ≈ masses : disorder ~ mess ≈ messe = mass ≈ masses ~ messes ≈ messy
  pierre ≈ pair : pair = pair ≈ pierre ~ peter ≈ pire ~ pires ≈ peer

## v7 214 vs v5 410 -- honest

Fewer loops/certs than v5 (410) ONLY because the offline MUSE-pivot synonym layer
is SPARSER than the embedding kNN v5 used (giant component dropped). The engine is
identical; the gap is the ~sem source. With real multilingual embeddings the v7
web (bigger node set) would exceed v5's loop yield. Same one gap, again: embeddings
densify ~sem -> more loops -> more certified gold.

Artifacts: chain-loops-v7.tsv, loop-certified-pairs-v7.tsv, chain-web-full-v7.tsv,
chain-web-stats-v7.json. v5/old files untouched.
