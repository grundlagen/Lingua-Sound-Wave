# Non-cognate homophonic writing from the hop dataset (no cheat)

The project IS non-cognate: FR sounds like EN, means something ELSE, both read
(Van Rooten). Cognate sound==meaning (group=groupe) is the trivial cheat -- excluded.

Generated from hops-all.tsv: for each EN word take ≈snd renders that are NOT its
translation (non-cognate), chain to maximise EN-bigram AND FR-bigram coherence.
Pure data, no invention.

  EN: two when tourist     FR: tout n triste
  EN: so small designs     FR: ce sol design
  EN: left lie an          FR: lift la in

Honest (Fable): structurally correct (non-cognate, sound-alike, derived) but THIN,
because word-by-word ≈snd can't re-cut boundaries -- it never reaches whole-stream
carve grade (Humpty Dumpty -> un petit un petit). The carve engine on the full
phoneme stream is the STRONG non-cognate method; this word-chain is the cheap one.
Bigram LM caps fluency. The fix is the same one gate: an LLM trained on the hop
dataset learns to (a) carve whole streams and (b) make the non-cognate FR read as
a sentence -- the two things word-chaining + bigram can't. So: dataset ready,
method ready, output thin until the L2 model arranges it.
