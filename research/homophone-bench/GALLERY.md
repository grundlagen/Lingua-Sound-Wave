# Gallery — LLM-approved homophonic lines (live Nemotron judge)

Lines composed by `bank_composer.py --llm` and scored for French coherence by a
real LLM (Nemotron via OpenRouter), not the bigram. Scores are honest: most
homophonic carves are *strained* French (0.1–0.4); the judge correctly rates the
rare natural one high. Each line: read the FR aloud and it lands on the EN.

## The standout (unseeded)

```
EN: to tell you not at work and while the door
FR: t elle en hâte avec un voile d or          [homophone 0.56, FR-coherence 0.85]
    ("un voile d'or" = a veil of gold — genuinely natural French)
```

## Themed harvest (seeded; LLM French-coherence shown)

```
night → d une des îles après n est fort son âme        FR 0.35
        ("des îles … son âme" = islands … his soul — night-apt)
war   → déroulement met êtes vous n en hâte avec un homme   FR 0.40
the   → d âme tout n est elle et oui dit à               FR 0.30
love  → les hommes des faunes des femmes ils mis en donne   FR 0.10
to    → tout où est fort les traitent les hommes et aides   FR 0.10
```

## Reading

The LLM judge is the **L2-model upgrade** working: it exposes that bigram-fluent
≠ French-coherent (it rates the old "fluent" carves 0.0–0.4) and reliably surfaces
the rare genuinely-French line. The ceiling now is candidate *supply*: the bank
must offer the judge more natural-French options. Two levers feed that — the
balanced French-anchored bank, and filtering bank units by the LLM itself
(`bank_llm_filter.py`) so only French-natural units remain to compose from.
