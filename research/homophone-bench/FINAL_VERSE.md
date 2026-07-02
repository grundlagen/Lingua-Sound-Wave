# Final verse — homophonic writing, sensible in both languages

Method: `final_verse.py` composes from STRICT tiles only (bank sound ≥ 0.68,
ladder sound ≥ 0.85, no identity/cognate tiles, real-word gate) → machine beams
propose; an LLM-in-the-loop (the missing L2-coherence model named in
SESSION_HANDOFF) selects and grammar-fixes; every line is then **verified by the
judge of record** (`matcher` combo) — the score printed is the machine's, not
taste. One phoneme stream, two readings.

## The gallery (combo = verified sound identity, 1.00 = same stream)

**combo 1.00**
> EN: *less debt, less mess*
> FR: *laisse dette, laisse messe* — "leave the debt, leave the mass"

**combo 0.95**
> EN: *do tell me, who said less?*
> FR: *doux, tel mie, où cède laisse?* — "soft, like crumb — where does the leash yield?"

**combo 0.87**
> EN: *we knew the sea*
> FR: *oui, nous, ceci* — "yes, us, this"

**combo 0.86**
> EN: *my movie, my mess*
> FR: *mes mous vies, mes messes* — "my soft lives, my masses"

**combo 0.85**
> EN: *bless the chef, bless the soup*
> FR: *blesse le chef, blesse la soupe* — "wound the chef, wound the soup"

**combo 0.84**
> EN: *sell the soup, seize the seat*
> FR: *selle la soupe, sise le site* — "saddle the soup, sited the site"

**combo 0.83**
> EN: *moo, said the moose; boo, said the fool*
> FR: *mou, cède la mousse; boue, cède la foule* — "soft, yields the foam; mud, yields the crowd"

**combo 0.81**
> EN: *less and less, the bell said: dawn*
> FR: *laisse en laisse, label cède, donne* — "leash on leash — the label yields, gives"

`bless/blesse` is the emblem: the English blesses, the French wounds — same
sound, opposite worlds. Van Rooten's move exactly.

## What made it work (vs the 0.31–0.64 machine-only lines)

1. **Strict tiles only.** Composing from sound ≥ 0.85 atoms means the line's
   homophony is inherited, not hoped for. The earlier composers averaged over
   weak tiles and produced glue-soup.
2. **LLM as the L2 model.** The bigram LM ranks word-adjacency, not sense — the
   standing ceiling in SESSION_HANDOFF. Claude selecting among verified tiles IS
   the missing coherence model; the judge (combo) stays the arbiter so taste
   can't inflate scores.
3. **Grammar templates that survive in both languages**: imperative + noun
   (laisse/blesse/selle X), apposition lists (oui, nous, ceci), quotative
   parallel (X, cède la Y).

## Repeatable recipe

```
python final_verse.py -n 10        # machine beams (tiles verified, both LMs)
# then: LLM selects/fixes grammar; re-verify each line with matcher combo;
# keep only combo >= 0.80. The verify snippet lives in FINAL_VERSE.md history.
```

Next lever: the DUAL tiles (translation ∧ homophone, `dual_mine.py`) make the
two readings *mean the same too* — from Van Rooten (same sound, different sense)
to full dual translation (same sound, same sense). Needs the full mine re-run
(container restart killed it at 17k rows).
