# The hop dataset = LLM training set for homophonic-semantic writing

## How writing is formulated from hops

A hop = one move: ≈ (sound) or ~ (meaning). A WORD render = a short hop-path
(root ≈ route ~ route ≈ routes). A SENTENCE = chain the per-content-word paths
into one surface that reads in both languages. So:

  writing = pick content -> route each unit through hops to a sound-true render
            -> concatenate renders into a line fluent on BOTH sides.

hop-train.jsonl: 16,719 examples {in, out, hops, path}. Format an LLM learns:
  input  = source word/meaning
  target = homophonic render + the hop PATH (chain-of-thought)
The path teaches the model WHY (sound vs meaning move), not just the pair.

## Your length point is right

word-level expansion = 1.01x (none). But LINE-level expands:
  Humpty Dumpty (2w) -> un petit un petit (4w)
  ...sat on a wall (6w) -> ...et on vol (7w)
Because reaching a sound-match needs FILLER hops (un, d', et) to spend the sound
budget. Homophonic translation is NOT length-preserving -- output grows by the
necessary intermediate/filler hops. The LLM must learn to spend that growth on
FLUENT fillers, not salad.

## Critical (Fable lens)

An LLM trained on these hops gets two things the deterministic stack lacked, in
one: (1) PATH SELECTION (which hop-sequence to emit) and (2) L2 COHERENCE (make
the expanded surface read as a sentence) -- the missing gate. The dataset already
encodes sound (≈) + meaning (~); the model learns to weave them fluently.
Caveat: train on the SOUND-true paths (verified by the matcher) so the model can't
hallucinate sound -- it only learns to ARRANGE verified hops by meaning/fluency,
exactly the iron rule (LLM never invents phonetics). Feed hop-train.jsonl + the
subway hubs (for shortest paths) + the carve lines (for expansion) -> an LLM that
writes sensical homophonic-semantic translation, expansion and all.
