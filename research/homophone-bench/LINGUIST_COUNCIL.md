# Linguistics council — meter & meaning improvement clues

_Saved advisories from multiple LLMs; review before encoding._


## DeepSeek

(unavailable)


## Gemini

(unavailable)


## Nemotron

**A. METER / PROSODY SCORING – concrete, codeable tweaks**

| # | Rule (what to add) | IPA / word illustration | How to encode (weight / gap / feature / model) |
|---|--------------------|--------------------------|-----------------------------------------------|
| 1 | **French phrase‑final prominence boost** – give extra credit when the last stressed syllable of the line aligns with a French *accent d’insistance* (usually the final lexical stress). | EN line ends on ˈkæt → FR line ends on ˈkat (chat) | Add **+0.3** to the prosody score if the final syllable carries a primary stress marker (ˈ) in the French phoneme string and the English line’s final stress falls within ±1 syllable. |
| 2 | **Syllable‑weight penalty for excess light syllables** – French verse prefers a mix of heavy (CVC) and light (CV) syllables; too many light syllables in a row hurt naturalness. | EN “the a b c” → FR “lə a b k” (all CV) | Scan the French phoneme string; for every run of ≥3 consecutive light syllables (no coda) subtract **0.15** per extra light syllable beyond the first two. |
| 3 | **E‑muet (schwa) count constraint** – in classical French meter, an e‑muet may be elided before a vowel; count eligible elisions and reward lines that achieve the target meter after elision. | EN “to be or not to be” → FR “tə bə ɔʁ nə tə bə” (schwas before vowels) | Compute potential elisions: each schwa (ə) followed by a vowel‑initial word yields a *possible* elision. If the line’s target syllable count (e.g., 8 for octosyllabic) is met after applying all possible elisions, add **+0.2**; otherwise apply a gap of **‑0.1** per missing syllable. |
| 4 | **Caesura/hemistich alignment** – French alexandrine expects a strong pause after the 6th syllable; English lines often have a natural break elsewhere. | EN “ˈʃeɪdz ˈɔːv ˈlaɪf” → FR “ʃe dz | ɔv laif” (| = caesura) | Detect a major stress boundary (primary stress or a pause >150 ms from TTS) around syllable 6±1. If present, add **+0.25**; if the boundary is >2 syllables away, subtract **‑0.2**. |
| 5 | **Secondary‑stress smoothing** – English secondary stress (ˌ) often maps to a French weak beat; penalize mismatches where a secondary stress falls on a French strong beat (phrase‑final or caesura). | EN ˌɡræsp → FR ɡʁas (secondary on onset) | For each secondary stress marker in the English phoneme string, check the French syllable’s position: if it coincides with a French strong beat (primary stress, phrase‑final, or caesura) apply a gap **‑0.15**; otherwise neutral. |
| 6 | **Rising vs. falling contour match** – French intonation rises toward phrase‑final prominence; English often falls. Compute a simple pitch‑slope feature from a lightweight F0 tracker (e.g., Praat‑like) on the synthesized French utterance and compare to the English source slope. | EN falling contour on “ˈhʌmpi ˈdʌmpti” → FR rising on “œ̃ pəˈti œ̃ pəˈti” | Extract normalized slope (ΔF0 / duration). If English slope < 0 (falling) and French slope > 0 (rising) add **+0.2**; if both have same sign, apply **‑0.1**. |
| 7 | **Onset‑salience boost for French‑friendly clusters** – reward onsets that are permissible in French (e.g., /st/, /ʁ/, /ɛk/) and penalize illegal clusters (e.g., English /θr/). | EN “θrɪŋ” → FR “tʁɛ̃” (illegal /θr/ → repaired) | For each onset, lookup a binary French‑onset table; add **+0.1** per legal onset, subtract **‑0.1** per illegal onset that required a repair. |

*Implementation note*: each rule contributes a scalar term; the final prosody score = Σ(weight × term) + baseline. Weights can be tuned on a held‑out set of human‑judged sound‑true translations.

---

**B. MEANING SCORING – concrete, codeable tweaks**

| # | Rule (what to add) | IPA / word illustration | How to encode (weight / gap / feature / model) |
|---|--------------------|--------------------------|-----------------------------------------------|
| 1 | **Collocation likelihood** – use a French bigram language model (e.g., 5‑gram from OpenSubtitles) to score the likelihood of each adjacent word pair in the output. | EN “grasp the chance” → FR “saisir l’opportunité” (bigram “saisir l’” high) | Compute log‑probability Σ log P(wᵢ|wᵢ₋₁). Normalize by line length; map to **[0,1]** via sigmoid; add **+0.3 × normScore**. |
| 2 | **Semantic coherence via sentence‑level embedding** – embed the French line with a lightweight multilingual SBERT (e.g., distiluse‑base‑multilingual‑cased) and compute cosine similarity to the embedding of the source English line (after translating via a pivot MT system for reference). | EN “Humpty Dumpty” → FR “un petit un petit” (low similarity) → penalize; better: “un œuf tombé” (high similarity) | Sim = cosine(embed_FR, embed_EN_ref). If Sim < 0.4 apply gap **‑0.2**; if Sim ≥ 0.6 add **+0.2**. |
| 3 | **Image‑concreteness bonus** – reward content words that score high on a French concreteness norm (e.g., from the French Lexicon Project). | FR “rivière” (concrete) vs. FR “justice” (abstract) | For each content noun/adjective, lookup concreteness (0‑5). Average across line; if avg ≥ 3.5 add **+0.15**; if ≤ 2.0 apply **‑0.1**. |
| 4 |
