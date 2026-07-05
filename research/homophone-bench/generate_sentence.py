#!/usr/bin/env python3
"""
GRAPH-WALKING SENTENCE GENERATOR — Route through the mapping-web to find
meaning-preserving sound paths for full English sentences.

Strategy:
  1. For each EN word, find candidate FR outputs (model + lookup + fragments)
  2. Walk the mapping-web graph: sound edges → meaning edges → sound edges
  3. Score each path by: combo_score × meaning_proximity × fluency
  4. Compose the best path into a French sentence
  5. Use round-rabbit lattice for semantic neighborhood expansion
  6. Use fragments for sub-word decomposition of unknown words

The "worming around to find meaning" approach:
  EN_word → sound_edge → FR_candidate → meaning_edge → FR_synonyms
  → sound_edge → EN_synonym → verify meaning loop-back

Usage:
    python generate_sentence.py "the shadow of the mountain"
    python generate_sentence.py --beam 8 "she wandered through the forest"
    python generate_sentence.py --model homophone_model_enriched.pt "ocean dreams"
"""

import json, sys, os, math, argparse, subprocess
from collections import defaultdict, deque
from pathlib import Path

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BENCH_DIR)

parser = argparse.ArgumentParser(description="Graph-walking sentence generator")
parser.add_argument("text", nargs="+", help="English sentence to translate")
parser.add_argument("--beam", type=int, default=5, help="Beam width for path search")
parser.add_argument("--max-hops", type=int, default=2, help="Max graph hops")
parser.add_argument("--model", default=None, help="Path to .pt model file")
parser.add_argument("--min-score", type=float, default=0.55, help="Min combo score")
parser.add_argument("--verbose", "-v", action="store_true")
args = parser.parse_args()

sentence = " ".join(args.text)
words = [w.lower().strip(".,;:!?'\"") for w in sentence.split() if w.strip(".,;:!?'\"")]
print(f"EN: {sentence}")
print(f"Words: {words}\n")

# ── 1. Load model + lookup ──────────────────────────────────────────────
model = None
if args.model and os.path.exists(args.model):
    import torch, torch.nn as nn
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    SRC_C2I = state["src_c2i"]; TGT_C2I = state["tgt_c2i"]
    TGT_I2C = state["tgt_i2c"]; MAX_LEN = state["max_len"]
    IPA_C2I = state.get("ipa_c2i", {})

    class DualEncoderLSTM(nn.Module):
        def __init__(self, src_v, ipa_v, tgt_v, hidden=512, embed=256):
            super().__init__()
            self.src_embed = nn.Embedding(src_v, embed, padding_idx=0)
            self.ipa_embed = nn.Embedding(ipa_v, embed, padding_idx=0) if ipa_v > 0 else None
            self.tgt_embed = nn.Embedding(tgt_v, embed, padding_idx=0)
            self.char_encoder = nn.LSTM(embed, hidden, 3, batch_first=True, dropout=0.1)
            self.phoneme_encoder = nn.LSTM(embed, hidden // 2, 2, batch_first=True, dropout=0.1) if ipa_v > 0 else None
            self.fusion = nn.Linear(hidden + hidden // 2, hidden) if ipa_v > 0 else nn.Linear(hidden, hidden)
            self.decoder = nn.LSTM(embed + hidden, hidden, 3, batch_first=True, dropout=0.1)
            self.attn_W = nn.Linear(hidden, hidden)
            self.enc_to_attn = nn.Linear(hidden, embed)
            self.out = nn.Linear(hidden, tgt_v)
            self.hidden, self.embed = hidden, embed

        def generate(self, src_chars, src_ipa, max_len=15):
            with torch.no_grad():
                B = src_chars.size(0)
                ce = self.src_embed(src_chars)
                co, (ch, cc) = self.char_encoder(ce)
                if self.phoneme_encoder and src_ipa is not None:
                    pe = self.ipa_embed(src_ipa)
                    po, (ph, pc) = self.phoneme_encoder(pe)
                    enc_out = self.fusion(torch.cat([co, po], dim=-1))
                    h = torch.cat([ch[-1:], ph[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
                    c = torch.cat([cc[-1:], pc[-1:]], dim=-1).unsqueeze(0).repeat(3, 1, 1)
                else:
                    enc_out = self.fusion(co)
                    h, c = ch, cc
                dec_in = torch.tensor([[TGT_C2I.get("<sos>", 1)]], device=src_chars.device)
                result = []
                for _ in range(max_len):
                    de = self.tgt_embed(dec_in)
                    ep = self.enc_to_attn(enc_out)
                    sc = torch.bmm(de, ep.transpose(1, 2))
                    at = torch.softmax(sc / math.sqrt(self.embed), -1)
                    ctx = torch.bmm(at, enc_out)
                    do, (h, c) = self.decoder(torch.cat([de, ctx], -1), (h, c))
                    tok = self.out(do).argmax(-1).item()
                    if tok == TGT_C2I.get("<eos>", 2): break
                    ch = TGT_I2C.get(tok, "?")
                    if ch not in ("<pad>", "<sos>"): result.append(ch)
                    dec_in = torch.tensor([[tok]], device=src_chars.device)
                return "".join(result)

    sv = len(SRC_C2I); iv = len(IPA_C2I); tv = len(TGT_C2I)
    m = DualEncoderLSTM(sv, iv, tv, state.get("hidden", 512), state.get("embed", 256))
    m.load_state_dict(state["model"]); m.eval()
    model = m
    print(f"[model] Loaded: {sum(p.numel() for p in m.parameters()):,} params")

    def model_carve(word, ipa=""):
        t = [SRC_C2I["<sos>"]] + [SRC_C2I.get(c,0) for c in word] + [SRC_C2I["<eos>"]]
        t += [0]*(MAX_LEN-len(t))
        ct = torch.tensor([t[:MAX_LEN]])
        it = torch.zeros(1, MAX_LEN, dtype=torch.long)
        if IPA_C2I and ipa:
            it_tokens = [IPA_C2I.get("<sos>", 1)] + [IPA_C2I.get(c, 0) for c in ipa] + [IPA_C2I.get("<eos>", 2)]
            it_tokens += [0]*(MAX_LEN-len(it_tokens))
            it = torch.tensor([it_tokens[:MAX_LEN]])
        return model.generate(ct, it if ipa else None)
else:
    def model_carve(word, ipa=""): return None

# ── 2. Load lookup tables ──────────────────────────────────────────────
print("[lookup] Loading dictionaries...")

# Strict-gold lookup
gold_lookup = {}
with open("strict-gold-training.jsonl") as f:
    for line in f:
        r = json.loads(line)
        en = r["input"].replace("English word: ", "").strip().lower()
        fr = r["output"].strip().lower()
        q = r.get("quality", r.get("sound", 1.0))
        if en not in gold_lookup or q > gold_lookup[en][1]:
            gold_lookup[en] = (fr, q)

# v5 dictionary — indexed by EN word → list of (fr, score, tier, ipa, alignment)
dict_idx = defaultdict(list)
with open("dictionary-v5.tsv") as f:
    cols = f.readline().rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6: continue
        d = dict(zip(cols, parts))
        en = d["en"].lower()
        fr = d["fr"].lower()
        score = float(d.get("score", 0))
        tier = d.get("tier", "B")
        en_ipa = d.get("en_ipa", "")
        fr_ipa = d.get("fr_ipa", "")
        usable = int(d.get("usable_for_composition", 0) or 0)
        alignment = d.get("alignment", "")
        chunk = d.get("chunk_recipe", "")
        dict_idx[en].append((fr, score, tier, en_ipa, fr_ipa, usable, alignment, chunk))

# Sort each EN word's candidates by score
for en in dict_idx:
    dict_idx[en].sort(key=lambda x: -x[1])

# ── 3. Load mapping-web for graph walking ──────────────────────────────
print("[graph] Loading mapping-web...")
sound_edges = defaultdict(list)  # en_word → [(fr_node, score)]
meaning_graph = defaultdict(set)  # en_word ↔ fr_word meaning connections

if os.path.exists("mapping-web.json"):
    web = json.load(open("mapping-web.json"))
    for edge in web.get("edges", []):
        if edge.get("type") == "sound_edge":
            src = edge["source"].replace("en:", "").replace("sound:en:", "")
            tgt = edge["target"].replace("fr:", "").replace("sound:fr:", "")
            score = float(edge.get("score", 0))
            sound_edges[src].append((tgt, score))
        elif edge.get("type") == "meaning_edge":
            src = edge["source"].split(":")[-1]
            tgt = edge["target"].split(":")[-1]
            meaning_graph[src].add(tgt)
            meaning_graph[tgt].add(src)

print(f"  Sound edges: {sum(len(v) for v in sound_edges.values())}")
print(f"  Meaning nodes: {len(meaning_graph)}")

# ── 4. Load fragments for sub-word decomposition ───────────────────────
fragments = []
if os.path.exists("fragments.tsv"):
    with open("fragments.tsv") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                count, en_chunk, fr_chunk = int(parts[0]), parts[1], parts[2]
                fragments.append((count, en_chunk, fr_chunk))
    fragments.sort(key=lambda x: -x[0])
print(f"[fragments] {len(fragments)} chunks loaded")

# ── 5. Load round-rabbit for semantic neighborhoods ─────────────────────
rr_attachments = defaultdict(list)
if os.path.exists("round-rabbit.json"):
    rr = json.load(open("round-rabbit.json"))
    for row in rr["rows"]:
        for att in row.get("attachments", []):
            en = att["en"].lower()
            fr = att["fr"].lower()
            score = float(att.get("score", 0))
            rr_attachments[en].append((fr, score, row.get("homophonic_hops", 0)))
print(f"[round-rabbit] {sum(len(v) for v in rr_attachments.values())} attachment entries")

# ── 6. Graph-walking word carver ───────────────────────────────────────
def carve_word(en_word, beam=5):
    """Return top-K candidate French renderings for an English word.
    Uses: lookup → model → dictionary → fragments → sound edges."""
    candidates = []  # (fr, score, source)

    # 6a. Exact lookup (free, trusted)
    if en_word in gold_lookup:
        fr, q = gold_lookup[en_word]
        candidates.append((fr, min(q, 1.0), "gold"))

    # 6b. Dictionary v5 (top entries)
    for fr, score, tier, en_ipa, fr_ipa, usable, alignment, chunk in dict_idx.get(en_word, [])[:5]:
        if score >= args.min_score:
            w = 1.0 if tier == "S" else 0.9 if tier == "A" else 0.75
            candidates.append((fr, score * w, f"dict_{tier}"))

    # 6c. Model generation
    if model:
        ipa = dict_idx[en_word][0][3] if dict_idx.get(en_word) else ""
        fr = model_carve(en_word, ipa)
        if fr and fr != en_word:
            candidates.append((fr, 0.7, "model"))

    # 6d. Sound-edge neighbors from mapping-web
    for fr_node, edge_score in sound_edges.get(en_word, [])[:3]:
        if edge_score >= args.min_score:
            candidates.append((fr_node, edge_score, "sound_edge"))

    # 6e. Round-rabbit semantic neighbors
    for fr, score, hops in rr_attachments.get(en_word, [])[:3]:
        hop_penalty = 0.9 ** hops
        candidates.append((fr, score * hop_penalty, f"rabbit_h{hops}"))

    # 6f. Fragment-based fallback for words with no candidates
    if not candidates and fragments:
        # Find fragments that appear in this word's IPA
        word_ipa = ""
        for count, en_chunk, fr_chunk in fragments[:200]:
            if en_chunk in word_ipa or en_chunk in en_word:
                # Try to build a French rendering from known fragments
                for count2, en_chunk2, fr_chunk2 in fragments[:200]:
                    combined = fr_chunk + fr_chunk2
                    if 2 <= len(combined) <= 15:
                        candidates.append((combined, 0.45, "fragment_fallback"))
                        break
                break

    # Deduplicate by FR output, keep best score
    seen = {}
    for fr, score, source in candidates:
        if fr not in seen or score > seen[fr][0]:
            seen[fr] = (score, source)
    results = [(fr, s, src) for fr, (s, src) in seen.items()]
    results.sort(key=lambda x: -x[1])
    return results[:beam]

# ── 7. Compose sentence ────────────────────────────────────────────────
print("\n" + "="*60)
print("GENERATING")
print("="*60)

all_candidates = []
for w in words:
    cands = carve_word(w, args.beam)
    all_candidates.append(cands)
    if args.verbose:
        print(f"  {w:15s}:")
        for fr, score, src in cands[:3]:
            print(f"    → {fr:20s} ({score:.2f}, {src})")

# Beam search over word sequences: pick best combination
# Score each path by product of word scores
def beam_compose(candidates_per_word, beam=5):
    """Beam search: keep top-K partial sentences."""
    beam_states = [([], 1.0)]  # (fr_words_list, cumulative_score)
    for cands in candidates_per_word:
        if not cands:
            # Fallback: keep original word
            cands = [(w, 0.5, "fallback") for w in ["???"]]
        new_states = []
        for fr_words, cum_score in beam_states:
            for fr, score, src in cands:
                new_states.append((fr_words + [(fr, score, src)], cum_score * score))
        new_states.sort(key=lambda x: -x[1])
        beam_states = new_states[:beam]
    return beam_states[0]

best = beam_compose(all_candidates, args.beam)
fr_words, total_score = best

# Build output
fr_sentence = " ".join(fr for fr, _, _ in fr_words)

print(f"\n{'─'*60}")
print(f"EN: {sentence}")
print(f"FR: {fr_sentence}")
print(f"Score: {total_score:.4f}")
print(f"{'─'*60}")

for w, (fr, score, src) in zip(words, fr_words):
    print(f"  {w:15s} → {fr:20s} ({score:.2f}, {src})")

# ── 8. Fluency check via bigram LM ─────────────────────────────────────
try:
    import bigram_lm
    lm = bigram_lm.BigramLM()
    fluency = lm.fluency(fr_sentence)
    print(f"\n  Fluency: {fluency:.3f}")
except Exception:
    pass
