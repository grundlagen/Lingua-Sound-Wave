#!/usr/bin/env python3
"""
BENCHMARK — Three sentence-level homophone modes.

MODE 1 (Word lookup): Per-word DB lookup. Baseline — we have this working.
    "the ocean remembers" → FR words from dict → Agent B hears English words.
    Fast, deterministic, no sentence-level coherence.

MODE 2 (Semantic + Homophonic): LLM rewrites the English sentence so that
    the resulting French BOTH sounds like English AND makes sense as French.
    Both input and output can change. Meaning preservation is a soft goal.
    "the ocean remembers" → "the sea recalls" → "la scient recalle"
    (French that an English ear hears as "the sea recalls" AND is valid French)

MODE 3 (Pure Homophone): Best possible sound match at sentence level.
    LLM generates French that an English speaker would hear as the original English.
    French coherence doesn't matter. Maximize Agent B score only.
    "the ocean remembers" → "zeau sheaux remembre" (gibberish French that
    sounds exactly like English)

SCORING:
    - Agent B score: How well an English ear hears the original English in the French
    - FR Fluency: How natural the French sentence is (bigram LM)
    - Semantic: How well the meaning is preserved (optional, via back-translation)

Run: python3 benchmark_sentence.py [--model MODEL] [--llm LLM]
"""

import json, os, sys, math, time, subprocess, argparse, re
from collections import defaultdict
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
BENCH_DIR = Path("/home/mint/Lingua-Sound-Wave/research/homophone-bench")
os.chdir(BENCH_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lookup",
                    choices=["lookup", "lstm", "transformer", "all"],
                    help="Which homophone model to use for generation")
parser.add_argument("--llm", default="deepseek",
                    choices=["deepseek", "none"],
                    help="LLM backend for sentence rewriting")
parser.add_argument("--test", default="all",
                    choices=["all", "quick"],
                    help="Test set size")
parser.add_argument("--mode", type=int, default=0,
                    choices=[0, 1, 2, 3],
                    help="Run single mode (0=all)")
args = parser.parse_args()

# ── 1. Load resources ───────────────────────────────────────────────────

# 1a. Lookup DB
print("[1/5] Loading homophone dictionary...")
with open("strict-gold-training.jsonl") as f:
    raw = [json.loads(line) for line in f]
lookup = {}
for r in raw:
    en = r["input"].replace("English word: ", "").strip().lower()
    fr = r["output"].strip().lower()
    q = r.get("quality", r.get("sound", 1.0))
    if en and fr and en != fr:
        if en not in lookup or q > lookup[en][1]:
            lookup[en] = (fr, q)
print(f"  {len(lookup)} EN→FR pairs loaded")

# 1b. Synonyms for English rewriting
syn_en = defaultdict(set)
if os.path.exists("muse-pivot-syn.tsv"):
    for line in open("muse-pivot-syn.tsv", encoding="utf-8"):
        a, b, _ = line.rstrip("\n").split("\t")
        if a.startswith("en:") and b.startswith("en:"):
            syn_en[a[3:]].add(b[3:])
            syn_en[b[3:]].add(a[3:])
    print(f"  {len(syn_en)} synonym sets loaded")

# 1c. English IPA dictionary for Agent B
en_ipa_dict = {}
if os.path.exists("en-word-ipa.tsv"):
    for i, line in enumerate(open("en-word-ipa.tsv", encoding="utf-8")):
        if i == 0: continue
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2 and p[1] and "(fr)" not in p[0]:
            en_ipa_dict[p[0].lower()] = p[1].replace(" ", "")
print(f"  {len(en_ipa_dict)} EN IPA entries")

# 1d. French LM for fluency
FR_LM = None
try:
    import bigram_lm
    import __main__
    # Fix pickle issue: BigramLM might reference __main__ if pickled from CLI
    if not hasattr(__main__, 'BigramLM'):
        __main__.BigramLM = bigram_lm.BigramLM
    FR_LM = bigram_lm.load("fr")
    print("  French bigram LM loaded")
except Exception as e:
    print(f"  [!] French LM not available: {e}")

# 1e. LSTM model (optional)
LSTM_MODEL = None
if args.model in ("lstm", "all"):
    try:
        import torch, torch.nn as nn
        state = torch.load("homophone_model_gpu.pt", map_location="cpu", weights_only=False)
        SRC_C2I = state["src_c2i"]; TGT_C2I = state["tgt_c2i"]
        TGT_I2C = state["tgt_i2c"]; MAX_LEN = state["max_len"]
        class LSTMHomophone(nn.Module):
            def __init__(self, sv, tv, h=512, e=256):
                super().__init__()
                self.src_embed = nn.Embedding(sv, e, padding_idx=0)
                self.tgt_embed = nn.Embedding(tv, e, padding_idx=0)
                self.encoder = nn.LSTM(e, h, 3, batch_first=True, dropout=0.1)
                self.decoder = nn.LSTM(e+h, h, 3, batch_first=True, dropout=0.1)
                self.attn_W = nn.Linear(h, h)
                self.enc_to_attn = nn.Linear(h, e)
                self.out = nn.Linear(h, tv)
                self.hidden, self.embed = h, e
            def generate(self, src, ml=15):
                with torch.no_grad():
                    se = self.src_embed(src)
                    eo, (h, c) = self.encoder(se)
                    di = torch.tensor([[TGT_C2I["<sos>"]]], device=src.device)
                    r = []
                    for _ in range(ml):
                        de = self.tgt_embed(di)
                        ep = self.enc_to_attn(eo)
                        sc = torch.bmm(de, ep.transpose(1, 2))
                        at = torch.softmax(sc / math.sqrt(self.embed), -1)
                        ctx = torch.bmm(at, eo)
                        do_, (h, c) = self.decoder(
                            torch.cat([de, ctx], -1), (h, c))
                        tok = self.out(do_).argmax(-1).item()
                        if tok == TGT_C2I["<eos>"]: break
                        ch = TGT_I2C.get(tok, "?")
                        if ch not in ("<pad>", "<sos>"): r.append(ch)
                        di = torch.tensor([[tok]], device=src.device)
                    return "".join(r)
        lstm = LSTMHomophone(len(SRC_C2I), len(TGT_C2I))
        lstm.load_state_dict(state["model"]); lstm.eval()
        LSTM_MODEL = lstm
        LSTM_SRC_C2I = SRC_C2I
        LSTM_TGT_C2I = TGT_C2I
        LSTM_TGT_I2C = TGT_I2C
        LSTM_MAX_LEN = MAX_LEN
        print("  LSTM model loaded")
    except Exception as e:
        print(f"  [!] LSTM not available: {e}")

# 1f. Transformer model (optional)
TRANSFORMER_MODEL = None
if args.model in ("transformer", "all"):
    try:
        import torch, torch.nn as nn
        state = torch.load("homophone_transformer.pt", map_location="cpu", weights_only=False)
        # Transformer config
        cfg = state.get("config", {})
        d_model = cfg.get("d_model", 512)
        nhead = cfg.get("nhead", 8)
        num_layers = cfg.get("num_layers", 6)
        SRC_C2I_T = state["src_c2i"]; TGT_C2I_T = state["tgt_c2i"]
        TGT_I2C_T = state["tgt_i2c"]; MAX_LEN_T = 18
        class TransformerHomophone(nn.Module):
            def __init__(self, sv, tv, d=512, nh=8, nl=6):
                super().__init__()
                self.d_model = d
                self.src_embed = nn.Embedding(sv, d, padding_idx=0)
                self.tgt_embed = nn.Embedding(tv, d, padding_idx=0)
                self.pos_encoder = nn.Embedding(500, d)  # simplified
                self.transformer = nn.Transformer(
                    d_model=d, nhead=nh, num_encoder_layers=nl,
                    num_decoder_layers=nl, dim_feedforward=d*4,
                    dropout=0.1, batch_first=True)
                self.out = nn.Linear(d, tv)
            def generate(self, src, ml=18):
                with torch.no_grad():
                    pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
                    se = self.src_embed(src) + self.pos_encoder(pos)
                    memory = self.transformer.encoder(se)
                    di = torch.tensor([[TGT_C2I_T["<sos>"]]], device=src.device)
                    r = []
                    for i in range(ml):
                        dp = torch.arange(1, device=src.device).unsqueeze(0)
                        de = self.tgt_embed(di) + self.pos_encoder(dp)
                        out = self.transformer.decoder(de, memory)
                        tok = self.out(out[:, -1:, :]).argmax(-1).item()
                        if tok == TGT_C2I_T["<eos>"]: break
                        ch = TGT_I2C_T.get(tok, "?")
                        if ch not in ("<pad>", "<sos>"): r.append(ch)
                        di = torch.cat([di, torch.tensor([[tok]], device=src.device)], 1)
                    return "".join(r)
        transformer = TransformerHomophone(
            len(SRC_C2I_T), len(TGT_C2I_T), d_model, nhead, num_layers)
        # Load state dict if available
        if "model" in state:
            transformer.load_state_dict(state["model"], strict=False)
        transformer.eval()
        TRANSFORMER_MODEL = transformer
        TRANSFORMER_SRC_C2I = SRC_C2I_T
        TRANSFORMER_TGT_C2I = TGT_C2I_T
        TRANSFORMER_TGT_I2C = TGT_I2C_T
        TRANSFORMER_MAX_LEN = MAX_LEN_T
        print(f"  Transformer model loaded (epoch {state.get('epoch','?')}, val_loss={state.get('val_loss','?')})")
    except Exception as e:
        print(f"  [!] Transformer not available: {e}")

# ── 2. Core functions ────────────────────────────────────────────────────

def tts(text, voice):
    """eSpeak IPA generation."""
    r = subprocess.run(
        ["espeak-ng", "-q", "--ipa", "-v", voice, text],
        capture_output=True, text=True)
    ipa = r.stdout.strip()
    for c in "ˈˌ": ipa = ipa.replace(c, "")
    return ipa

def ndice(a, b, n=2):
    """N-gram Dice coefficient."""
    A = {a[i:i+n] for i in range(len(a)-n+1)} if len(a) >= n else {a}
    B = {b[i:i+n] for i in range(len(b)-n+1)} if len(b) >= n else {b}
    return 2*len(A&B)/(len(A)+len(B)) if (A or B) else 1.0

def agent_B(fr_word):
    """What English word does a French utterance sound like?"""
    en_ipa = tts(fr_word, "en-us").replace(" ", "")
    best_en, best_s = fr_word, 0.0
    for en_w, en_w_ipa in list(en_ipa_dict.items())[:3000]:
        s = ndice(en_ipa, en_w_ipa)
        if s > best_s:
            best_s, best_en = s, en_w
    return best_en, best_s

def agent_B_phrase(fr_phrase):
    """Score: how well does the whole French phrase sound like the English target?"""
    en_ipa = tts(fr_phrase, "en-us").replace(" ", "")
    return en_ipa

def fr_fluency(fr_text):
    """Score how natural the French text is. Returns 0-1."""
    if FR_LM is None:
        return 1.0  # no LM available
    words = [w.strip(".,;:!?\"'") for w in fr_text.lower().split() if w.strip(".,;:!?\"'")]
    if len(words) < 2:
        return 0.5
    return FR_LM.fluency(words)

def generate_word(word, model_type="lookup"):
    """Generate French for a single English word."""
    if model_type == "lookup" or model_type == "none":
        if word in lookup:
            return lookup[word][0]
        # Fuzzy match
        best = max(lookup.keys(), key=lambda k: len(set(k)&set(word))/max(1, len(set(k)|set(word))))
        return lookup[best][0]
    elif model_type == "lstm" and LSTM_MODEL:
        t = [LSTM_SRC_C2I["<sos>"]] + [LSTM_SRC_C2I.get(c, 0) for c in word] + [LSTM_SRC_C2I["<eos>"]]
        t += [0] * (LSTM_MAX_LEN - len(t))
        return LSTM_MODEL.generate(torch.tensor([t[:LSTM_MAX_LEN]]))
    elif model_type == "transformer" and TRANSFORMER_MODEL:
        t = [TRANSFORMER_SRC_C2I["<sos>"]] + [TRANSFORMER_SRC_C2I.get(c, 3) for c in word] + [TRANSFORMER_SRC_C2I["<eos>"]]
        t += [0] * (TRANSFORMER_MAX_LEN - len(t))
        return TRANSFORMER_MODEL.generate(torch.tensor([t[:TRANSFORMER_MAX_LEN]]))
    return word

def llm_rewrite(prompt, max_tokens=200):
    """Call LLM for sentence rewriting."""
    # Try DeepSeek API via environment
    api_key = os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        # Fallback: return None (caller should handle)
        return None
    try:
        import urllib.request
        url = "https://api.deepseek.com/v1/chat/completions"
        data = json.dumps({
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }).encode()
        req = urllib.request.Request(url, data=data, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return None

# ── 3. Three Modes ───────────────────────────────────────────────────────

def mode1_word_lookup(sentence, model="lookup"):
    """Mode 1: Per-word lookup/generation. No sentence-level awareness."""
    words = [w.lower().strip(".,;:!?\"'()") for w in sentence.split()
             if w.strip(".,;:!?\"'()")]
    fr_words = []
    b_words = []
    scores = []
    for w in words:
        fr = generate_word(w, model)
        fr_words.append(fr)
        heard, hs = agent_B(fr)
        b_words.append(heard)
        scores.append(hs)
    fr_text = " ".join(fr_words)
    he_text = " ".join(b_words)
    avg_b = sum(scores) / max(1, len(scores))
    fluency = fr_fluency(fr_text)
    return {
        "en_original": sentence,
        "en_changed": sentence,  # unchanged in mode 1
        "fr": fr_text,
        "heard": he_text,
        "b_score": avg_b,
        "fluency": fluency,
        "per_word": list(zip(words, fr_words, b_words, scores))
    }

def mode2_semantic_homophonic(sentence, model="lookup"):
    """
    Mode 2: Sentence-level semantic + homophonic.
    LLM rewrites English to maximize homophone quality, then generates French.
    Both input and output are coherent sentences.
    """
    # Step 1: Find which words need better homophones
    words = [w.lower().strip(".,;:!?\"'()") for w in sentence.split()
             if w.strip(".,;:!?\"'()")]
    weak = []
    strong = []
    for w in words:
        if w in lookup:
            fr, q = lookup[w]
            if q >= 0.7:
                strong.append((w, fr, q))
            else:
                weak.append((w, fr, q))
        else:
            # Check synonyms for homophone coverage
            syns = syn_en.get(w, set())
            best_syn, best_fr, best_q = None, None, 0
            for syn in syns:
                if syn in lookup and lookup[syn][1] > best_q:
                    best_q = lookup[syn][1]
                    best_syn, best_fr = syn, lookup[syn][0]
            if best_syn and best_q > 0.5:
                strong.append((best_syn, best_fr, best_q))
            else:
                weak.append((w, generate_word(w, model), 0.3))

    # Step 2: Build LLM prompt to rewrite the English sentence
    weak_words = [w for w, _, _ in weak]
    strong_pairs = [(w, fr) for w, fr, q in strong if q >= 0.7]

    prompt = f"""Rewrite this English sentence so that every word has a strong French homophone match.
The rewritten sentence must be natural, grammatical English that preserves the original meaning.

ORIGINAL: {sentence}

Words that already have good homophones: {strong_pairs[:8]}
Words needing better matches: {weak_words[:8]}

RULES:
- You may replace any word with a synonym or paraphrase
- The result must be a coherent English sentence
- Prefer common, phonetically rich words (they have more French homophone options)
- Keep the same general meaning and tone

Return ONLY the rewritten sentence, nothing else."""

    rewritten = llm_rewrite(prompt)
    if not rewritten:
        # Fallback: use synonym substitution
        new_words = []
        for w in words:
            if w in weak_words and w in syn_en:
                syns = syn_en[w]
                best = max(syns, key=lambda s: lookup[s][1] if s in lookup else 0, default=w)
                new_words.append(best if best in lookup else w)
            else:
                new_words.append(w)
        rewritten = " ".join(new_words)
    
    # Step 3: Generate French from rewritten English
    rw_words = [w.lower().strip(".,;:!?\"'()") for w in rewritten.split()
                if w.strip(".,;:!?\"'()")]
    fr_words = []
    b_words = []
    scores = []
    for w in rw_words:
        fr = generate_word(w, model)
        fr_words.append(fr)
        heard, hs = agent_B(fr)
        b_words.append(heard)
        scores.append(hs)
    
    fr_text = " ".join(fr_words)
    he_text = " ".join(b_words)
    avg_b = sum(scores) / max(1, len(scores))
    fluency = fr_fluency(fr_text)
    
    return {
        "en_original": sentence,
        "en_changed": rewritten,
        "fr": fr_text,
        "heard": he_text,
        "b_score": avg_b,
        "fluency": fluency,
        "per_word": list(zip(rw_words, fr_words, b_words, scores))
    }

def mode3_pure_homophone(sentence, model="lookup"):
    """
    Mode 3: Best possible homophone at sentence level.
    Maximize Agent B score. French coherence is irrelevant.
    English can be rewritten. The goal: an English ear hears the original.
    """
    words = [w.lower().strip(".,;:!?\"'()") for w in sentence.split()
             if w.strip(".,;:!?\"'()")]
    
    # For each word, find the absolute best-sounding French out of ALL candidates
    fr_words = []
    b_words = []
    scores = []
    
    for w in words:
        candidates = []  # (fr_word, b_score)
        
        # Direct lookup
        if w in lookup:
            fr, q = lookup[w]
            _, hs = agent_B(fr)
            candidates.append((fr, hs))
        
        # Try all FR words from the DB that phonetically resemble this EN word
        # (sample 500 random FR words for speed)
        sample_fr = list(lookup.values())[:500]
        for fr, _ in sample_fr:
            _, hs = agent_B(fr)
            candidates.append((fr, hs))
        
        # Try model-generated candidates
        if model != "lookup":
            for _ in range(3):  # try a few variations
                fr = generate_word(w, model)
                if fr and fr != w:
                    _, hs = agent_B(fr)
                    candidates.append((fr, hs))
        
        # Pick best
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            fr_words.append(best[0])
            scores.append(best[1])
            b_words.append(agent_B(best[0])[0])
        else:
            fr_words.append(w)
            scores.append(0.0)
            b_words.append(w)
    
    fr_text = " ".join(fr_words)
    he_text = " ".join(b_words)
    avg_b = sum(scores) / max(1, len(scores))
    fluency = fr_fluency(fr_text)
    
    return {
        "en_original": sentence,
        "en_changed": sentence,  # unchanged in pure homophone mode
        "fr": fr_text,
        "heard": he_text,
        "b_score": avg_b,
        "fluency": fluency,
        "per_word": list(zip(words, fr_words, b_words, scores))
    }

# ── 4. Benchmark ─────────────────────────────────────────────────────────

print(f"\n[2/5] Model: {args.model}  |  LLM: {args.llm}")
print("=" * 70)

if args.test == "quick":
    tests = [
        "the ocean remembers every vessel",
        "she walked through the silent forest",
    ]
else:
    tests = [
        "the ocean remembers every vessel that ever sailed",
        "she wandered through the silent forest at twilight",
        "a gentle stream becomes a mighty rushing river",
        "the shadow of the mountain falls across the water",
        "light breaks where no sun shines",
        "their words had the weight of stones",
        "the clock struck one and the silence deepened",
        "beyond the hills the sky turned gold and red",
    ]

modes_to_run = [args.mode] if args.mode else [1, 2, 3]

results = []

for sent in tests:
    print(f"\n{'─'*70}")
    print(f"EN: {sent}")
    sent_results = {"en": sent, "modes": {}}
    
    for mode_num in modes_to_run:
        t0 = time.time()
        
        if mode_num == 1:
            label = "MODE 1 (word lookup)"
            r = mode1_word_lookup(sent, args.model)
        elif mode_num == 2:
            label = "MODE 2 (semantic+homophonic)"
            r = mode2_semantic_homophonic(sent, args.model)
        elif mode_num == 3:
            label = "MODE 3 (pure homophone)"
            r = mode3_pure_homophone(sent, args.model)
        
        elapsed = time.time() - t0
        
        sent_results["modes"][mode_num] = r
        sent_results["modes"][mode_num]["time"] = elapsed
        
        print(f"\n  {label}")
        if mode_num == 2 and r["en_changed"] != sent:
            print(f"  EN′: {r['en_changed']}")
        print(f"  FR : {r['fr']}")
        print(f"  HE : {r['heard']}")
        print(f"  B-score: {r['b_score']:.3f}  |  FR-fluency: {r['fluency']:.4f}  |  time: {elapsed:.1f}s")
        
        # Per-word detail
        for w, fr, he, hs in r["per_word"][:10]:
            bar = "█" * int(hs * 10) + "░" * (10 - int(hs * 10))
            print(f"    {w:12s} → {fr:15s} → B:{he:12s} [{bar}] {hs:.2f}")
        if len(r["per_word"]) > 10:
            print(f"    ... ({len(r['per_word']) - 10} more words)")
    
    results.append(sent_results)

# ── 5. Summary ───────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("SUMMARY — Average scores across all test sentences")
print(f"{'='*70}")
print(f"{'Mode':<30s} {'B-score':>8s} {'Fluency':>8s} {'Time':>8s}")
print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*8}")

for mode_num in [1, 2, 3]:
    if mode_num not in modes_to_run:
        continue
    b_scores = [r["modes"][mode_num]["b_score"] for r in results if mode_num in r["modes"]]
    fluencies = [r["modes"][mode_num]["fluency"] for r in results if mode_num in r["modes"]]
    times = [r["modes"][mode_num]["time"] for r in results if mode_num in r["modes"]]
    
    avg_b = sum(b_scores)/max(1, len(b_scores))
    avg_f = sum(fluencies)/max(1, len(fluencies))
    avg_t = sum(times)/max(1, len(times))
    
    names = {1: "MODE 1 (word lookup)", 2: "MODE 2 (semantic+homophonic)", 3: "MODE 3 (pure homophone)"}
    print(f"{names[mode_num]:<30s} {avg_b:>8.3f} {avg_f:>8.4f} {avg_t:>7.1f}s")

print(f"\n  B-score = how well an English ear hears the original")
print(f"  Fluency = how natural the French sentence is")
print(f"  Higher is better for both.")
print(f"  Mode 2 balances both. Mode 3 maximizes sound alone.")

# Save detailed results
out_path = "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nDetailed results saved to {out_path}")
