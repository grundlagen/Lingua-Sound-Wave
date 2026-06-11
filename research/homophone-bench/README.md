# Homophone Bench v5

Composition-ready EN/FR homophone material for Lingua Sound Wave.

## Commands

Use the local virtualenv created for this bench:

```bash
.venv/bin/python merge_generative.py
.venv/bin/python function_glue.py
.venv/bin/python finalize.py
.venv/bin/python compose_lots.py
.venv/bin/python recursive_poet.py
.venv/bin/python round_rabbit.py --seed "west air breeze rose"
.venv/bin/python mapping_web.py
.venv/bin/python smoke_test.py
```

Equivalent CLI after editable install:

```bash
.venv/bin/pip install -e .
homophone-bench merge-generative
homophone-bench function-glue
homophone-bench finalize
homophone-bench compose-lots
homophone-bench recursive-poet
homophone-bench round-rabbit --seed "west air breeze rose"
homophone-bench mapping-web
homophone-bench smoke
```

## Outputs

- `dictionary-v5.json` and `dictionary-v5.tsv`: reviewed dictionary plus validated generated entries.
- `composition-index.json`: indexes by pivot, first/final class, syllables, direction, tier, usability, and source stage.
- `composition-lots.json`: partial/whole/multi pattern lots for composition.
- `composition-lines.json` and `composition-lines.tsv`: deterministic dual-line smoke compositions with QC gates.
- `recursive-poem.json` and `recursive-poem.tsv`: recursive semantic-sound poem candidates.
- `round-rabbit.json` and `round-rabbit.tsv`: semantic components expanded by homophonic hop radius with all attached string substitutions.
- `mapping-web.json` and `mapping-walks.tsv`: typed sound, fragment, and meaning graph with permutation walks.
- `muse-status.json`: local check for a MUSE/Nemotron reference file.

## Release Rule

Generated fragment matches are not treated as hand gold. `merge_generative.py`
preserves `generator_score`, recomputes `independent_score`, fills `fr_ipa`,
enriches alignments and junctions, and stamps `source_stage` plus `accepted_by`.

Core function-word glue is also explicit. `function_glue.py` only marks or adds
composition-only rows, currently `the -> de` and existing `and -> end`, and the
finalizer requires an independent score of at least `0.55` before such glue can
be used by the composer.
