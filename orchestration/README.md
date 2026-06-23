# Fugu-style Orchestration Setup ("the Japanese-style AI mechanism")

This directory sets up a **Sakana Fugu-style multi-agent orchestrator** — a
coordinator model that assigns *Thinker / Worker / Verifier* roles to a pool of
LLMs, picks a topology, verifies, and does bounded recursive retry.

It is deliberately self-contained and isolated from the Lingua-Sound-Wave
linguistics code. Nothing here imports or touches the audio/phoneme stack.

---

## What's actually real (verified 2026-06-23)

I checked this rather than trusting the earlier chat, because the chat admitted
parts were unverifiable. Findings:

| Thing | Status | Weights? |
|---|---|---|
| **Sakana Fugu / Fugu Ultra** (official) | Real, launched **2026-06-22** | ❌ Closed. API-only, OpenAI-compatible endpoint. No weights. |
| **TRINITY** (arXiv 2512.04695, ICLR 2026) | Real paper. ~19.5K-param evolved router | Sakana's are closed; reimpl trains locally (gradient-free, runs anywhere) |
| **Conductor** (arXiv 2512.04388, ICLR 2026) | Real paper. RL-trained coordinator | Sakana's closed; **open reimpl weights exist** ↓ |
| **OpenFugu** (`trotsky1997/OpenFugu`) | 3rd-party open reimplementation | ✅ Runnable. Trains TRINITY + Conductor, OpenAI-compatible server |
| **`di-zhang-fdu/openfugu-conductor-3b`** | 3rd-party Conductor weights on HF | ✅ **Downloadable.** GRPO fine-tune of Llama-3.2-3B-Instruct, safetensors BF16 (~6 GB) |

**The official Fugu has no weights to hunt.** The only genuinely downloadable
"weights" are the community OpenFugu Conductor (a 3B Llama fine-tune) and the
TRINITY router, which you train yourself in minutes because it's tiny and
gradient-free.

## The honest caveat you need before spending bandwidth

Fugu's whole value is orchestrating **frontier worker models** (Opus, Gemini,
GPT). The coordinator weights you can download are the *cheap* part — the
~19.5K-param router or a 3B conductor. They are nearly worthless without a pool
of strong workers behind them.

So there are two regimes:

1. **Real Fugu-quality** = coordinator (local, cheap) + **frontier API workers**
   (paid, per-call). This is the architecture that works. It is *not* free and
   *not* fully local.
2. **Fully local / Termux** = coordinator + small local workers via Ollama
   (Llama/Gemma/DeepSeek). You get the *mechanism* — role assignment, topology,
   verify, retry — but small-model quality, not frontier quality. Good for
   learning the architecture and for cheap decomposable tasks; it will not
   match Fugu Ultra.

> Termux note: a 3B BF16 conductor is ~6 GB and won't run comfortably on a
> phone. On Termux, prefer the **TRINITY router** (negligible RAM) plus remote
> API workers, or a quantized GGUF conductor. See `pull-weights.sh`.

---

## Quick start

```bash
cd orchestration

# 1. Hunt + download the real artifacts (clones OpenFugu, pulls Conductor weights)
./pull-weights.sh

# 2. Bootstrap a venv, install deps, run the self-test
./setup-fugu.sh

# 3. Configure your worker pool (local Ollama and/or API keys)
cp workers.example.yaml .fugu/workers.yaml
$EDITOR .fugu/workers.yaml

# 4. Serve the OpenAI-compatible endpoint and point Claude Code / any client at it
cd .fugu/OpenFugu && python openfugu/serve.py
```

Everything heavy (the cloned repo, the multi-GB weights, the venv) lands under
`orchestration/.fugu/`, which is git-ignored — weights never get committed.

## Using it from Claude Code

OpenFugu serves `/v1/chat/completions`. Point any OpenAI-compatible client at
`http://localhost:8000/v1` and use it as a single model named `fugu`. Claude
Code itself stays your driver; Fugu becomes one more callable model in the pool.

## Sources

- Fugu launch / release: https://sakana.ai/fugu-release/
- Technical report: https://arxiv.org/abs/2606.21228
- TRINITY: https://arxiv.org/abs/2512.04695 · Conductor: https://arxiv.org/abs/2512.04388
- Official (closed): https://github.com/SakanaAI/fugu
- OpenFugu (open reimpl): https://github.com/trotsky1997/OpenFugu
- Conductor weights: https://huggingface.co/di-zhang-fdu/openfugu-conductor-3b
