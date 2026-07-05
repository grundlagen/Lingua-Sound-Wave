# Applied the Fable lens: real meaning edges (MUSE), not surface cognates

Critique was: web/chains' meaning hops are surface cognates (route~route), shallow.
Applied: fetched open MUSE en-fr dict (113k pairs), added real translation edges to
the v7 web.

meaning_edges 1,728 (cognate-only) -> 7,584 (+5,856 MUSE)  = 4.4x denser, real.

Now routes carry true sense+sound (en word -> real FR translation -> sounds like):
  cat   ~means~ chat  ~sounds~ en:chat     (means cat AND sounds like chat)
  peace ~means~ paix  ~sounds~ en:per

Artifact: mapping-web-v7-muse.json (v5/v7 webs untouched). round_rabbit runs on it.

End goal: this is the meaning-layer unlock -- chains/Round Rabbit now route real
semantics, not shared spelling. Next: embeddings for GRADED meaning (synonym-near,
not just exact translation) -> "sounds like a near-synonym of the translation",
the full homophonic+semantic translation engine. MUSE = exact bridge; embeddings =
the gradient.
