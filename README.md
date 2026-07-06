# Lingua-Sound-Wave

> ## ⭐ **[PIPELINE.md](PIPELINE.md)** — the canonical map of the homophonic generator (stages 1–5, engines, training, build order). Read it first.

**Transform Language into Sound Waves** — A TypeScript-powered project for converting textual and linguistic input into dynamic audio visualizations and soundscapes. Also home to the **homophonic-translation pipeline** (EN↔FR sound+meaning engine) and **fugu-swarm** (Fugu-style multi-agent orchestration).

## Overview

Lingua-Sound-Wave bridges linguistics, audio processing, and creative coding. Input text, poems, code, or natural language, and watch it transform into beautiful, responsive sound waves, waveforms, and generative audio experiences. Built with TypeScript for type-safe, modern web and Node.js compatibility.

Perfect for:
- Interactive art installations
- Educational tools for language learning through sound
- Music production aids (text-to-melody)
- Data sonification projects
- Accessibility tools (visual + auditory text representation)

See `research/homophone-bench/README.md` for the homophonic engine and pipeline documentation.

## Features (Current & Planned)

### Current
- Text parsing and phoneme analysis
- Real-time waveform visualization using Web Audio API / Canvas
- Basic frequency mapping from word length/syllables
- Export to WAV/MP3

### Planned New Stuff
- AI-powered semantic audio generation (integrate with local LLMs or APIs for emotion-based sound design)
- Multi-language support with IPA (International Phonetic Alphabet) to sound mapping
- 3D sound wave visualizations (Three.js)
- Collaborative sound wave remixing
- Plugin system for custom sonification rules
- Mobile app companion (React Native)
- Integration with MIDI for music production

## Getting Started

```bash
git clone https://github.com/grundlagen/Lingua-Sound-Wave.git
cd Lingua-Sound-Wave
npm install
npm run dev
```

Open in browser and start typing text to see/hear the magic!

## Tech Stack
- TypeScript
- Web Audio API
- Canvas/WebGL for visuals
- Node.js for backend processing (optional)
- Vite or Next.js for frontend

## Project Structure
```
Lingua-Sound-Wave/
├── src/
│   ├── core/           # Phoneme parser, frequency mapper
│   ├── audio/          # Sound generators, oscillators
│   ├── visual/         # Waveform renderers
│   └── utils/          # Helpers
├── public/
├── tests/
├── package.json
└── README.md
```

## How It Works
1. Parse input text into linguistic units (words, syllables, phonemes)
2. Map to audio parameters (frequency, amplitude, timbre, duration)
3. Generate real-time audio + synchronized visuals
4. Allow user interaction (sliders for intensity, speed, style)

## Contributing
We welcome contributions! Especially new sonification algorithms, UI improvements, and integrations.

See [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon)

## License
MIT

## Roadmap
- [x] Basic text-to-wave prototype
- [ ] Advanced AI integration
- [ ] 3D visuals
- [ ] Multi-lang support
- [ ] Mobile release

---

# fugu-swarm (sub-project)

A **no-ChatGPT, Fugu-style multi-agent orchestration layer** — one
OpenAI-compatible endpoint that routes each query across a swappable pool of
frontier models (Claude, Gemini, DeepSeek, Qwen, GLM, Kimi), with
Thinker/Worker/Verifier roles and bounded retry.

It is a thin layer over **[OpenFugu](https://github.com/trotsky1997/OpenFugu)**
(the engine), plus a curated worker pool that hard-excludes OpenAI/GPT, and an
adaptive bandit router borrowed from the `fugue` swarm. Nothing third-party is
copied or redistributed — `setup.sh` fetches it from licensed sources.

> Built on independent reimplementations of Sakana AI's Fugu. Not affiliated
> with Sakana AI.

---

## Why this exists / what's real

Sakana **Fugu** (launched 2026-06-22) is sold as one model but is really a
*policy over models*: a tiny coordinator routes each query to a pool of frontier
LLMs and returns one answer. Fugu's own weights are **closed (API-only)**. The
runnable, open pieces are reimplementations. I checked all four:

| Repo | What it is | Use it? |
|---|---|---|
| **trotsky1997/OpenFugu** | Python, ~3.8k LOC, Apache-2.0. Faithful: TRINITY router (verified **95%/100%** vs the real released checkpoint), Conductor DAG executor, GRPO training, OpenAI-compatible server, litellm pool. | ✅ **The engine.** fugu-swarm builds on it. |
| **nshkrdotcom/trinity_coordinator** | Elixir/Nx, 60★, 302 tests. | ↪ Upstream that *produced* the router weights OpenFugu fetches. Use only if you want an Elixir runtime. |
| **BicaMindLabs/open-sakanafugu** (`fugue`) | Shell/JS coding swarm: 9 LLM implementers + reviewer, bounded review-fix loop, Beta-Bernoulli bandit routing. | ↪ Idea source for our bandit. ⚠️ Its reviewer is **Codex (OpenAI)** — swap it to honour "no ChatGPT". |
| **Sakana-AI-labs/Sakana-Fugu** | Empty dirs, fake desktop installers, copies official branding. | ⛔ **Avoid** — impersonation stub; don't run the installers. |

### The honest caveat
Fugu's value comes from the **frontier workers**, not the coordinator. The
coordinator you run locally is tiny (~19.5K params for TRINITY); the quality
lives in the API workers behind it. So this is *not* a free, fully-local
frontier model — it's a smart router in front of paid APIs (or smaller local
models, at smaller-model quality).

## The mechanism in one breath

A ~0.6B backbone (Qwen3-0.6B) never answers you. It produces one hidden state;
a bias-free linear head scores 7 worker-slots + 3 roles (Worker/Thinker/
Verifier); the top worker is dispatched and *its* reply is returned. The loop
runs until a Verifier ACCEPTs or max-turns. **Fugu-Ultra** swaps the per-turn
picker for a Conductor that emits a whole workflow DAG. No worker weights are
ever touched — it's macro-composition over other people's models.

## What fugu-swarm adds

- **`pool/no-chatgpt.yaml`** + **`fugu_swarm/pool.py`** — the 7-slot pool with a
  *hard-enforced* no-OpenAI/GPT/Codex invariant (`slot_csv` raises if a ChatGPT
  model slips in). Strong models carry the Verifier role.
- **`fugu_swarm/bandit.py`** — a Beta-Bernoulli Thompson-sampling selector that
  learns which workers actually pass verification for your workload. Original
  code, offline-testable, no GPU.
- **`fugu_swarm/run.py`** — thin launcher: yaml → enforced pool → OpenFugu
  `serve.py`.

## Quickstart

```bash
./setup.sh                               # vendor OpenFugu, fetch artifacts, install deps
make test                                # offline unit tests (no models needed)

export ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  DEEPSEEK_API_KEY=...   # etc.
python -m fugu_swarm.run                 # preview the serve command (+ env preflight)
python -m fugu_swarm.run --serve         # launch the OpenAI-compatible endpoint :8088

curl localhost:8088/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"flatten a nested list in one line"}]}'
```

Point Claude Code (or any OpenAI-compatible client) at `http://localhost:8088/v1`
and call the model `fugu`. For a **fully offline** pool, swap any
`provider/model` line in the yaml for `ollama/<model>` and start `ollama serve`.

## Wiring the bandit (optional)

`serve.py`'s `Coordinator` picks the worker slot from the TRINITY head. To bias
that by observed success, subclass it and consult the bandit:

```python
from fugu_swarm.bandit import BanditRouter
b = BanditRouter([f"slot{i}" for i in range(7)], seed=0)
# in your dispatch: agent_id = int(b.select(candidates)[-1]) ; ... ; b.update(f"slot{agent_id}", accepted)
```

Keep it a *bias*, not a replacement — the head encodes per-query routing the
bandit can't see.

## Termux / low-RAM

Run the **TRINITY** coordinator (negligible RAM) with **remote API workers** —
don't try to host a 3B Conductor or local 8B workers on a phone. The router is
the only thing that must run locally.

## Layout

```
fugu_swarm/   bandit.py · pool.py · run.py        # our code (tested)
pool/         no-chatgpt.yaml                      # the worker pool
tests/        test_bandit.py · test_pool.py        # offline, 10 passing
setup.sh      vendor OpenFugu + fetch artifacts
vendor/       OpenFugu (git-ignored, fetched)      # the engine
```

## Note on this repo's hosting

This was generated inside a Claude Code session scoped to another repo, where
creating a new GitHub repo was not permitted. It lives on an **orphan branch**
(no shared history) as transport. To lift it into its own repo:

```bash
git clone -b fugu-swarm --single-branch <this-repo-url> fugu-swarm
cd fugu-swarm && rm -rf .git && git init
git add . && git commit -m "Initial commit: fugu-swarm"
git remote add origin <your-new-repo-url> && git push -u origin main
```

## License

Apache-2.0 (`LICENSE`). Third-party material is fetched, not redistributed; see
`NOTICE`.
