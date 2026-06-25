# web_poet — generative homophonic poetry by themed walk

The webbing result said: don't *translate a fixed line* (both rails co-chain ≈ 0).
So `web_poet.py` flips the problem — it **generates**. It walks the v7 web so the
poem is whatever sound-continuous, on-theme ribbon already lives in the
dictionary, and never has to force two fixed sentences to align.

## How the walk writes a poem

- **Strict typed alternation** — a *sound* hop (`≈`, the homophonic hand-off that
  keeps the line as one continuous stream when read aloud) then a *meaning* hop
  (`~`/`=`, which steers the sense). Typed, never the "scrabble" free mix.
- **Theme gravity** — meaning hops are pulled toward a seed theme vector
  (embeddings), so the poem drifts *around* a subject rather than wandering off.
- **Dual atoms as rests (★)** — landing on a loop-certified pair (where sound ==
  meaning) is rewarded; those are the cadence points, the line's natural stresses.
- Every word is a real, frequency-checked EN/FR node; every junction is a real
  matcher/graph edge. Nothing is invented.

Read the ribbon aloud in a French mouth and the `≈` hand-offs make **one
continuous sound**; read it as words and it is a found poem inside the lexicon.

## Selected output (real runs)

```
THEME: night
  lit · let · ★lettres · ★letters · ★laid aise · ★laid in · ★letting · laisse · les · lasse
  (one /l/-vowel stream: bed→let→letters→laid→letting; "lit" = FR 'bed', theme-apt)

THEME: sea
  shore · ★chauds · ★chaude · showed · ★seen · ★si · scie · saw · saut · sauter · ★sortie
  (a continuous sh/s tide drifting shore→seen→see→saw→leap→exit)

  sea · ★si · scie · saw · saut · saute · soaked · wet · ouest · ★western · ★ouais tonnes
  (sea→saw→sauté→soaked→wet→west)

THEME: love
  dear · dis · di · ★dit · ★cet · ★set · ★sets · ★cette · this · dix · ten
  (one d/s/t stream; rests on dit / cet / cette)

THEME: dream
  ★destiny · ★destined · destiné · destinés · debt inner · ★debtor · ★der · ★se · si
  (dream drifts to DESTINY and holds there — theme gravity at work, 7 rests)

THEME: star
  starred · ★stand · ★standing · stade · stage · met scène · medicine · médecine
  (star→stand→stadium→STAGE→scene — star-adjacent sense, found by the walk)
  ★mark · ★marques · ★max · ★maximum · madame · maîtresse · mattress · ★bed · ★baie
  (mark→max→madame→mistress→mattress→bed→bay)

THEME: war
  battle · bat elle · gras elle · gravel · dirt · dette · ★debt · ★dettes · ★seas · ★sexes
  (battle→gravel→dirt→debt — one continuous b/g/d/t stream)

THEME: death
  grave · gré ive · ★ive · ★eve · ève · of · ★de · ★der · ★se · si · so
  (grave→Eve→of→de… EN-flow 0.57; "grave" is the theme-apt opening)
```

## Why this is the right creative escape

The translation framing demanded both readings be parallel *and* coherent *and*
homophonic — three constraints that the data shows almost never co-occur.
Generation keeps only two (sound-continuous + on-theme) and lets the third
(coherence) be supplied by ranking, so the machine always returns *something*
real, and the dual atoms give it spine. The remaining lever is unchanged: swap
the bigram fluency for a real L2 model and the same walks become verse rather
than sound-true word-strings.

Run: `python web_poet.py sea night love`
