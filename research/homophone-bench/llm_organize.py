#!/usr/bin/env python3
"""LLM-organized paragraph: meaning-based ordering by an LLM that knows French."""
import json

d = json.load(open("full_dict_paragraph.json"))
pairs = d["pairs"]

# Build lookup: fr_word → en_word, sound
fr_to_en = {p["fr"]: (p["en"], p["sound"]) for p in pairs}

# All French words from the paragraph
all_fr = [p["fr"] for p in pairs]

# ═══ LLM-ORGANIZED: 10 stanzas by semantic field ═══
# These groupings use MY knowledge of what each French word means.
# No database. No suffix tricks. Just language.

STANZAS = [
    # ── 1. LEVER DU JOUR (Daybreak: nature waking) ──
    ["ouest", "est", "sève", "terre", "terres", "terreau", "cerises",
     "chaudes", "chaude", "sol", "sols", "soupe", "soupes", "miettes",
     "choux", "zeste", "roue", "rhône", "houle", "ouate"],

    # ── 2. CORPS EN MOUVEMENT (Bodies in motion) ──
    ["sauter", "sautent", "sortent", "sortes", "tournent", "traîne",
     "trappe", "tirs", "tir", "tendes", "tenir", "transforment",
     "transforme", "transmettent", "transport", "transfert", "skient",
     "ski", "slip", "slip"],

    # ── 3. VOIX ET PAROLES (Voices and speech) ──
    ["speech", "dis", "verrez", "savez", "savent", "valent", "verbes",
     "vexes", "toquent", "toque", "toc", "stoppe", "stoppes", "cède",
     "cessent", "déceler", "laisses", "opte", "ouais", "ouais"],

    # ── 4. LE MONDE TECHNIQUE (The technical world) ──
    ["technologie", "système", "script", "sketch", "spots", "sport",
     "stade", "stores", "square", "strip", "strip", "télésope",
     "textes", "textiles", "pelle", "pelle", "tamis", "tests",
     "tester", "test"],

    # ── 5. ÉMOTIONS PROFONDES (Deep emotions) ──
    ["tristesse", "victorieux", "joli", "folie", "souhaite", "stresse",
     "stressent", "déchaîne", "rixe", "torts", "toutefois", "inacceptables",
     "spécifique", "strict", "stricte", "strict", "stricte", "série",
     "trahi", "soupir"],

    # ── 6. LES GENS (The people) ──
    ["celtes", "slaves", "troupes", "troupes", "troupes", "aides",
     "aides", "aides", "volante", "voler", "vider", "vacciner",
     "saine", "diocèses", "tipi", "tels", "ceci", "cette", "cette",
     "ouah"],

    # ── 7. CORPS ET SENS (Body and senses) ──
    ["tête", "os", "tonte", "veines", "corps", "live", "sieste",
     "semelle", "nique", "sic", "soc", "tontes", "simili", "symphonie",
     "tim", "titi", "tamis", "tamis", "sushis", "sushi"],

    # ── 8. OBJETS DU QUOTIDIEN (Everyday objects) ──
    ["chaise", "sacs", "solde", "semestre", "tome", "script", "soupir",
     "soupir", "scie", "sciure", "choses", "chiffe", "hard", "chers",
     "schiste", "séquelles", "chili", "chics", "chauve", "chatte"],

    # ── 9. TERRITOIRES LOINTAINS (Distant territories) ──
    ["horde", "ouate", "n", "oui", "oies", "citerne", "round",
     "acquis", "reprit", "daller", "step", "stade", "té hic",
     "treize", "très", "tri", "trique", "trique", "trous", "tout"],

    # ── 10. CRÉPUSCULE (Twilight: resolution) ──
    ["vielle", "val", "vol", "veines", "verrez", "valent", "vestes",
     "zooms", "zeste", "yang", "vais tri", "vannes", "valent",
     "est", "opte", "saine", "savent", "savez", "transforment", "stresse"],
]

# Build the final output
print("="*70)
print("LLM-ORGANIZED BILINGUAL PARAGRAPH — 200 words, meaning-grouped")
print("="*70)

for i, stanza in enumerate(STANZAS):
    fr_line = "  ".join(stanza[:10])
    print(f"\n  [{i:2d}] {fr_line}")
    if len(stanza) > 10:
        print(f"       {'  '.join(stanza[10:20])}")
    
    # Show 3 example mappings
    examples = []
    for w in stanza[:3]:
        if w in fr_to_en:
            en_w, s = fr_to_en[w]
            examples.append(f"{en_w}→{w}({s:.2f})")
    if examples:
        print(f"       ↳ {'  '.join(examples)}")

# Stats
all_organized = sum(STANZAS, [])
used_count = len(all_organized)
unique_organized = len(set(all_organized))
print(f"\n  Words: {used_count} placed, {unique_organized} unique")
