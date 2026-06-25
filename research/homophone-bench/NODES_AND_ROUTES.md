# Nodes & routes: far-meaning transfer chains + any-word routing

Two capabilities built on the unedited v7 web (195,270 nodes), answering
"better chains that don't loop to the same meaning?" and "understand the
nodes / routes of connection to any word?"

## 1. Far-meaning transfer chains (`transfer_distance.py`)

Loops certify pairs whose meaning *returns home* — safe, but biased to
near-cognate morphology (`feel~files`). The opposite, and the actual art, is the
chain that stays **sound-bridged while traveling far in meaning**. We rank the
12,170 transfer chains (`chain-web-v7u.tsv`) by

```
score = sound_quality × (1 − cos(seed, endpoint))      # same MiniLM embeddings
```

so sounds-alike-but-means-something-different floats to the top. **815 transfers
land at cos < 0.2** (genuinely distant meaning). Top of the list:

```
leadership → chips     leadership ≈ lient chipe ~ … ≈ friendship ~ … ≈ chips
sandwich  → enseignants  sandwich ≈ sain dit chaise ~ … ≈ teachers ~ enseignants
veteran   → cuisinier    veteran ≈ fait reine ~ … ≈ patron ~ chefs ≈ chef = cuisinier
beef → bee · porgy → pizza · arrested → tweet · fries → free
```

Contrast set (the loop-like tail) is just identical words echoing
(`cause→cause`, `if→if`) — confirming the score separates *transfer* from *echo*.
Full ranking in `transfer-ranked-v7u.tsv` (sound × semantic-distance).

## 2. Routing & node understanding (`routes.py`)

Loads the cached graph and answers three things about **any** word in the system.

**`route A B`** — best alternation path (Dijkstra on −log quality). The canonical
example resolves exactly:

```
key → argile :  en:key ~ fr:clé ≈ en:clay = fr:argile      [3 hops, mean-q 0.92]
```

**`node W`** — connectivity profile. e.g. `key`: degree 12 (1 sound, 11 meaning);
sound twin `fr:qui`; meaning neighbours `clé/clef/touche/llave`; reach 11→52→193→
589 nodes over 1–4 hops; hub rank 1640/9964. `sea`: 6 sound twins
(`si/scie/ci/sis/…`), `= mer`, reach 699 by 4 hops.

**`hubs`** — global interchanges (nodes most chains pass through): `fr:dis`,
`fr:sain`, `fr:rient`, `fr:haines`, `fr:un`, `fr:sexe`, `fr:dit` … the
high-frequency French fragments that act as the web's transfer stations (the
"subway backbone").

### Honest limit

Under *strict alternation* with a 12-hop budget, some far pairs (`love→guerre`,
`sea→fromage`) return no route — that is the game constraint, not a connectivity
gap. Raw reachability is proven separately (`PROOFS_REACH_AND_COVERAGE.md`): any
word reaches any word over the unconstrained web.
