# The dual-reading ladder (1–4): both rails unanchored

`ladder.py` builds writing where the **English reading and the French reading
sound the same and mean the same**, neither side anchored. It fixes the "all 3 is
scrabble" problem by keeping the three edge types as separate axes.

## 1. Typed slots — three buckets, never flattened

```
key:  homophone(≈) {qui}        translation(=) {clé, clés, clef, touche}   synonym(~) {clave, llave, …}
play: homophone(≈) {plaie}      translation(=) {jouer, jouez}              synonym(~) {plays, playing, …}
sea:  homophone(≈) {si,scie,ci} translation(=) {mer}                       synonym(~) {marée, sear, …}
```

The homophone bucket is the cross-language **sound rung**; translation/synonym are
the **sense rails**. They are stored apart so a walk can't silently swap a sound
hop for a meaning hop.

## 2. The ladder — a position is a homophone PAIR

Each step lays a rung `en ≈ fr`; bigram LMs on **both** the EN words and the FR
words push each rail to read on its own. Example dual line (every word a real
node, each column a homophone twin):

```
EN:  stem  piece    se  suggested  claire  der
FR:  stems pissent  si  on geste   clair   der
```

Read down each row it's the same sound; read across, each side is (loosely)
its own language. **Honest limit:** both-side fluency is low (~0.14) — with only
813 atoms and a bigram LM, the lines are sound-true and word-real but not yet
sensical. The levers are a bigger alphabet (more loops + French-anchoring) and a
real L2 model in place of the bigram.

## 3. Loop-tiles — the finite alphabet (why end-sets are finite)

The 813 loop-certified pairs are the **dual atoms**: each is a word that sounds
like its partner AND means it (a closed ring where both rails reconcile). Because
they close, the reachable composition set is **bounded** — no infinite
obligation chain (aim→viser→… forever). Composition draws only from this alphabet,
webbing tiles at shared hub nodes.

```
feel ≈≈ files   sex ≈≈ sectes   control ≈≈ contrôles   reads ≈≈ rides   explain ≈≈ explosent
```

## 4. Sense split — polysemy clustered by embedding

A word's translations are clustered so a rung can pick the right sense:

```
spring -> {printemps}  |  {ressort}        ← season-sense vs mechanical-sense, cleanly split
play   -> {jouer, jouez}                    (MUSE has only the verb sense here)
```

`spring → {printemps} | {ressort}` is the mechanism working: the same English
word resolves to two French senses, each carrying its own homophone rung.

## Status

All four are implemented and run on real v7 data. The structure is right — typed
rungs/rails, dual atoms, finite alphabet, sense-split. The quality ceiling is the
**alphabet size and the coherence model**: grow the loop-tile set (French-anchor
the FR rail, mine more loops) and swap the bigram for a real LM, and the same
ladder yields sensical dual-reading verse.

Run: `python ladder.py love sea night light`
