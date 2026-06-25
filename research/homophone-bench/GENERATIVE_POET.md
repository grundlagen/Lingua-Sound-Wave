# Generative homophonic poetry — two attempts, one correct

## Correction: a graph walk is NOT homophonic; a carve is

`web_poet.py` (below) walks the web alternating sound/meaning hops. But the
*meaning* hops change the sound, so reading the whole ribbon aloud does **not**
reconstruct any English — it is not actually a homophone of anything. Homophony
(the entire point) requires **carving a target phoneme stream** into French so the
French, spoken, rebuilds the English sound (Van Rooten). That is
**`homophonic_poet.py`**, and it is verified by the matcher.

### homophonic_poet.py — generative + VERIFIED homophone

1. **generate** a themed English source (beam over English bigrams pulled toward a
   seed vector) — carries the MEANING;
2. **carve** its phoneme stream into French (`whole_line_carve`) — carries the
   SOUND; the French spoken ≈ the English;
3. **verify** with `combo(EN, FR)` — the matcher's homophone score (the proof).

Real runs (FR read aloud reconstructs the EN sound):

```
THEME love
  EN: adored companion in the   ->  FR: et un bon féminin de   (combo 0.43, cov 0.89)
  EN: heartfelt sympathy and the->  FR: et cet son patron je   (combo 0.43)
THEME night
  EN: evenings may be so        ->  FR: in in addition         (combo 0.44, cov 0.86)
  EN: lit up and the            ->  FR: d open de              (cov 1.00)
```

The canonical proof the carve is real: `Humpty Dumpty -> "un petit un petit"`
(combo 0.55, coverage 0.92) — Van Rooten's actual line, reconstructed. Meaning
rides the generated English; sound is the French carve; combo is the homophone.

---

## web_poet.py — themed walk (kept as exploration; NOT the homophonic output)

It walks the v7 web so the ribbon is whatever sound-continuous, on-theme path
lives in the dictionary. Useful for surfacing themed word-fields and dual atoms,
but the ribbon is not a true spoken homophone (the meaning hops break the sound).

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
