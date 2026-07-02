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
to full dual translation (same sound, same sense). **Mine complete: 102,899 dual
pairs; ladder 118,260** (DUAL-S 9,231, 272 non-cognate).

---

## The paragraph (one phoneme stream, prose in both languages)

Every sentence machine-verified: `combo` (sound identity), `prosody` (the
meter/stress channel — EN stress-timed vs FR syllable-timed divergence),
trigram L2 fluency. **Paragraph mean combo 0.84; prosody 0.90–1.00 throughout.**

> **EN** — We knew the sea. Less and less, the bell calls the fool; the queen
> sleeps at the pool. My movie, my mess. Do tell me, who said less? Bless the
> chef, bless the soup. Moo, said the moose; boo, said the fool. And at dawn,
> we do too. Less debt, less mess.

> **FR** — Oui, nous, ceci. Laisse en laisse, label colle la foule; la couine
> slip à la poule. Mes mous vies, mes messes. Doux, tel mie, où cède laisse ?
> Blesse le chef, blesse la soupe. Mou, cède la mousse ; boue, cède la foule.
> Et à donne, oui doux toux. Laisse dette, laisse messe.

*(FR gloss: "Yes, us, this. Leash on leash, the label glues the crowd; the
squeal slips to the hen. My soft lives, my masses. Soft, like crumb — where
does the leash yield? Wound the chef, wound the soup. Soft, yields the foam;
mud, yields the crowd. And in the deal, yes — sweet cough. Leave the debt,
leave the mass.")*

| sentence | combo | prosody | L2 |
|---|---|---|---|
| we knew the sea | 0.87 | 0.96 | 0.72 |
| less and less, the bell calls the fool | 0.75 | 0.93 | 0.35 |
| the queen sleeps at the pool | 0.75 | 0.92 | 0.39 |
| my movie, my mess | 0.86 | 0.90 | 0.27 |
| do tell me, who said less? | 0.95 | 0.95 | 0.25 |
| bless the chef, bless the soup | 0.85 | 0.95 | 0.39 |
| moo, said the moose; boo, said the fool | 0.83 | 0.96 | 0.22 |
| less debt, less mess | 1.00 | 1.00 | 0.37 |
| and at dawn, we do too | 0.80 | 0.93 | 0.40 |

## L2 upgrade (the standing ceiling, addressed)

`trigram_lm.py`: stupid-backoff trigram on 18.7M tokens of OpenSubtitles French
(1.66M trigrams). Sanity: *le chat mange la souris* 0.52 vs scrambled 0.37.
`final_verse.py` auto-uses it when `trigram-lm-fr.pkl` exists (50 MB, not
committed — rebuild: fetch OPUS fr.txt.gz slice, `python trigram_lm.py build fr
/tmp/fr_sub.txt`).

## Meter / prosody channel

The "language-difference" matching code the project built is `prosody.py`
(stress-weighted alignment, DIVERGED mode: English stress-timed & falling vs
French syllable-timed & phrase-final) — now used above as the second
verification channel alongside combo. `juncture.py` covers the cross-word
liaison/elision. Round-rabbit's need-word lattice is superseded at scale by
dual-pairs (hop-1) + chain-web (hop≥2), as suspected.
