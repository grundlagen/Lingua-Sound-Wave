import json
pairs = {}
for line in open("stage1_homophones.jsonl"):
    r = json.loads(line)
    if r.get("source") != "van-rooten":
        pairs[(r["en"], r["fr"])] = r
for i,line in enumerate(open("zipf-glue.tsv")):
    if i==0: continue
    p = line.rstrip("\n").split("\t")
    if len(p) >= 4:
        en, fr, snd = p[0], p[1], float(p[2])
        if (en,fr) not in pairs:
            pairs[(en,fr)] = {"en":en,"fr":fr,"sound":snd,"meaning":0.5,"source":"zipf-glue","tier":"F"}
with open("stage1_homophones.jsonl","w") as f:
    for (en,fr),d in sorted(pairs.items()): f.write(json.dumps(d,ensure_ascii=False)+"\n")
sources = {}; [sources.update({d["source"]:sources.get(d["source"],0)+1}) for d in pairs.values()]
print(f"Clean Stage 1: {len(pairs)} pairs"); [print(f"  {s}: {c}") for s,c in sorted(sources.items())]
