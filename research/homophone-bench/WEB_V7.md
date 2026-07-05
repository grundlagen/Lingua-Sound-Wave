# v7 web + Round Rabbit (Fable's engine, updated v5->v7)

Ran Fable's mapping_web.py + round_rabbit.py on dictionary-v7-integrated (16,208)
instead of v5. v5 and all old code/files kept intact (outputs saved as *-v7.*).

v7 web: 17,841 nodes / 25,342 edges  (v5: 15,681 / 19,120).
edges: sound 16,104 | fragment 2,640 | surface 3,106 | loanword 1,764 | meaning 1,728.

Round Rabbit on v7 (bigger field): air<-air/error/heir/hair; acoustic->egouts tic;
actes<-activist/oct/act/doctor; antique->en tic; adorable->et rebelles.

Artifacts: mapping-web-v7.json, round-rabbit-v7.json/tsv, mapping-walks-v7.tsv.
Run: feed dictionary-v7-integrated.json as the input mapping_web.py reads, in a
scratch dir (it writes mapping-web.json), then round_rabbit.py.

Critical (Fable lens): meaning edges still only from cognate flags (1,728, sparse,
no MUSE) -- the web is sound-dominant. To reach the end goal (homophonic+semantic
translation), the meaning layer must grow: MUSE/multilingual embeddings -> dense
meaning_edges -> Round Rabbit themes that actually steer content. Sound side is
now big and good; meaning side is the gap.
