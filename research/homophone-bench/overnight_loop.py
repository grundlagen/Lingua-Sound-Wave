#!/usr/bin/env python3
"""Overnight periphrastic generator — runs continuously, saves every 500 phrases."""
import json, os, random, time

os.chdir("/home/mint/Lingua-Sound-Wave/research/homophone-bench")

# Load Stage 1 lookup
lookup = {}
for line in open("stage1_homophones.jsonl",encoding="utf-8"):
    r = json.loads(line)
    lookup[r["en"]] = r["fr"].split()[0]

seeds = [
    "the sea remembers every ship","we call to the moon and she answers",
    "my sorrow sleeps in a deep well","bless the dawn that made us free",
    "less debt less mess more soup","the cat sat on the mat",
    "mary had a little lamb","she walks in beauty like the night",
    "the wind whispers through the trees","a gentle rain falls on the garden",
    "the stars shine bright above the mountain","love is a flame that never dies",
    "the old man walked along the shore","time flows like an endless river",
    "her voice was soft as evening light","the city sleeps beneath the snow",
    "a rose by any other name","to be or not to be",
    "all the world is a stage","the course of true love never did run smooth",
]

round_n = 0
while True:
    round_n += 1
    results = []
    for line in seeds * 30:
        words = [w.lower().strip(".,;:!?'\"") for w in line.split() if len(w)>=2]
        fr_parts = [lookup.get(w, f"«{w}»") for w in words]
        fr_text = " ".join(fr_parts)
        if fr_text.lower() != line.lower():
            results.append({"en": line, "fr": fr_text, "round": round_n})
    
    with open("stage3_overnight.jsonl","a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    total = sum(1 for _ in open("stage3_overnight.jsonl"))
    print(f"[{time.strftime('%H:%M:%S')}] Round {round_n}: {len(results)} phrases, total={total}")
    time.sleep(30)
