# Cross-word juncture re-scoring — the liaison/elision the judge forgets

Continuation of request #2 ("the elision/schwa/liaison rules that judges may
forget"). `rule_aware.py` covered the *within-word* performance rules (flapping,
l-vocalization, h-dropping, th-fronting, schwa-drop). `ELISION_PROPOSAL.md`
("Closing the loop") flagged what was still missing: the rules that fire **across
the word boundary** — liaison, elision-before-vowel, enchaînement, h-aspiré
block — which a matcher scoring word units **in isolation** cannot see.

## The gap, demonstrated

espeak applied to a whole phrase already does French liaison; applied word by
word it does not:

```
"les amis"     whole-phrase -> le-zami     (liaison /z/ surfaces)
"les" + "amis" word-by-word -> le-  + ami   (the /z/ is LOST)
```

The decoder / `phrase_weave` compose a French carve from a **word-keyed trie** and
score each unit's citation form. So a carve like *les amis* is scored as
/le.ami/, never the /lezami/ it becomes when spoken — and is **under-rated**
against a vowel-initial English source. Same for elision: the decoder emits the
unit `le` before a vowel (it can never spell `l'ami`), so espeak alone keeps
/lə.ami/; only a symbolic rule recovers /lami/.

## `juncture.py`

Reconstructs the connected-speech realization of a French word **sequence** and
scores the best realization against the source — the same honest upper-envelope
as `rule_aware` (every realization is a legal pronunciation, so it can only RAISE
a true homophone, never invent one). Two routes, max over both:

1. **espeak whole-phrase** — accurate where orthography licenses the sandhi.
2. **symbolic juncture pass** over per-unit citation IPA, applying:
   - **elision** — `le/de/je/me/te/se/ce/ne/que` drop final /ə/, `la` drops /a/,
     before a vowel-initial word (→ `l'ami` /lami/);
   - **liaison** — a latent word-final consonant surfaces before a vowel:
     `s,x,z → [z]`, `t,d → [t]`, `n → [n]`, `p → [p]`, `r → [ʁ]`, `g → [k]`,
     `f → [v]`, but only when it is genuinely silent in citation form (no double
     /t/ in *sept ans*, no double /n/ in *bonne amie*);
   - **h-aspiré block** — `le héros`, `les haricots` take neither (small lexical
     set, since the spelling alone can't tell aspiré from muet).

Enchaînement is intentionally **not** modelled: it only resyllabifies an
already-pronounced consonant, so it leaves the segment string unchanged.

## Result (`python juncture.py`)

```
EN source    FR carve           cite  +junc   gain
they zon     deux ans           0.36   0.39  +0.03   z-liaison
voozavay     vous avez          0.64   0.72  +0.08   z-liaison
noozom       nous hommes        0.72   0.82  +0.10   z-liaison
petit tammy  petit ami          0.82   0.88  +0.06   t-liaison
lammy        le ami             0.72   0.79  +0.07   elision (symbolic only)
la mee       la amie            0.72   0.79  +0.07   elision (symbolic only)
dunn ami     de un ami          0.63   0.67  +0.03   elision
  mean lift +0.049   (7/10 carves raised)
```

The elision rows lift **only** via the symbolic pass — espeak keeps /lə.ami/
because the carve is spelled `le ami`, not `l'ami`. That is exactly the value
the symbolic layer adds over re-running espeak.

## Wired into generation

`phrase_weave.transfer(..., juncture=True)` (CLI: `--juncture`) adds the sandhi
**lift** — measured with a single scorer, citation vs connected-speech, so it is
metric-consistent and only ever raises — to each candidate's sound score before
the joint re-rank. Default is **off**: existing behaviour is unchanged.

```
$ python phrase_weave.py --juncture "they zon"
  FR: des âne   snd 0.96 (was 0.93)   joint 0.81
  FR: les âne   snd 0.94 (was 0.91)   joint 0.80
```

## Verdict

This is the cross-word complement to `rule_aware`: together they cover the
within-word and at-the-boundary connected-speech rules that a citation-form judge
forgets. Consistent with `REPRESENTATION.md` — the dictionary stays broad-phoneme;
the sandhi is applied at match time, never committed to storage.
