"""Labeled EN<->FR benchmark pairs for homophone-matching method comparison.

Ground truth comes from linguistic knowledge, NOT from any scoring method:
  - word positives: dictionary-documented cross-lingual (near-)homophones;
  - phrase positives: concatenations of verified word units, plus documented
    homophonic translations (Van Rooten's "Mots d'Heures: Gousses, Rames",
    the "mayday" < "m'aider" etymology, anglicized "say la vee");
  - hard negatives: translation pairs (same meaning, unrelated sound) —
    this is the discrimination that actually matters for mining;
  - random negatives: shuffled unrelated pairings.

Each entry: (english_text, french_text, label, tier)
  label: 1 = sounds alike, 0 = does not
  tier:  "strong" | "loose" (positives) / "translation" | "random" (negatives)
"""

POS_WORDS = [
    # en, fr — strong, dictionary-verifiable near-identical pronunciations
    ("shoe", "chou"), ("key", "qui"), ("sea", "si"), ("knee", "ni"),
    ("bow", "beau"), ("toe", "tôt"), ("dough", "dos"), ("low", "l'eau"),
    ("foe", "faux"), ("sue", "sous"), ("do", "doux"), ("coo", "cou"),
    ("moo", "mou"), ("day", "dé"), ("shay", "chez"), ("mare", "mère"),
    ("sell", "selle"), ("bell", "belle"), ("tell", "telle"), ("wee", "oui"),
    ("fee", "fit"), ("pooh", "pou"), ("oh", "eau"), ("tray", "très"),
    ("bay", "baie"), ("ray", "raie"), ("gay", "gai"), ("seal", "cil"),
    ("peak", "pique"), ("beak", "bic"), ("say", "ses"), ("may", "mai"),
]

POS_WORDS_LOOSE = [
    # near-homophones with one mismatched feature (diphthong vs monophthong
    # already covered above; these add nasal vowels / final consonants)
    ("tan", "temps"), ("cone", "cône"), ("new", "nous"), ("sew", "sot"),
    ("mode", "mode"), ("cafe", "café"),
]

POS_PHRASES = [
    # concatenations of independently-verified units (sound-alike by construction)
    ("shoe key", "chou qui"),
    ("low toe", "l'eau tôt"),
    ("do say", "doux ses"),
    ("bell mare", "belle mère"),
    ("tell mare", "telle mère"),
    ("coo coo", "coucou"),
    ("may day", "m'aider"),          # documented etymology of "mayday"
    ("say la vee", "c'est la vie"),  # standard anglicization
    ("mercy", "merci"),
    ("oh low", "eau l'eau"),
]

POS_PHRASES_LOOSE = [
    # documented homophonic translation (loose, literary)
    ("Humpty Dumpty sat on a wall", "Un petit d'un petit s'étonne aux Halles"),
    ("Humpty Dumpty had a great fall", "Un petit d'un petit ah degrés te fallent"),
]

NEG_TRANSLATIONS = [
    # same meaning, phonetically unrelated (cognate-sounding pairs excluded)
    ("dog", "chien"), ("house", "maison"), ("bread", "pain"),
    ("cheese", "fromage"), ("book", "livre"), ("tree", "arbre"),
    ("moon", "lune"), ("fire", "feu"), ("snow", "neige"),
    ("winter", "hiver"), ("garden", "jardin"), ("window", "fenêtre"),
    ("apple", "pomme"), ("honey", "miel"), ("water", "eau"),
    ("bird", "oiseau"), ("horse", "cheval"), ("king", "roi"),
    ("queen", "reine"), ("night", "nuit"), ("morning", "matin"),
    ("thunder", "tonnerre"), ("shadow", "ombre"), ("silver", "argent"),
    ("gold", "or"), ("milk", "lait"), ("knife", "couteau"),
    ("chair", "chaise"), ("cloud", "nuage"), ("wave", "vague"),
]

NEG_TRANSLATION_PHRASES = [
    ("open the door", "ouvre la porte"),
    ("in the rain", "sous la pluie"),
    ("by the sea", "au bord de la mer"),
    ("good night", "bonne nuit"),
    ("I love you", "je t'aime"),
    ("let it be", "laisse faire"),
    ("over the moon", "aux anges"),
    ("with all my heart", "de tout mon cœur"),
    ("after the rain", "après la pluie"),
    ("through the woods", "à travers bois"),
]

NEG_RANDOM = [
    ("mountain", "pluie"), ("feather", "jardin"), ("candle", "cheval"),
    ("ladder", "fromage"), ("marble", "oiseau"), ("harbor", "tonnerre"),
    ("pencil", "vague"), ("ribbon", "couteau"), ("mirror", "ombre"),
    ("letter", "nuage"), ("cellar", "reine"), ("summer", "miel"),
    ("anchor", "fenêtre"), ("velvet", "matin"), ("copper", "chaise"),
]


def all_pairs():
    pairs = []
    for en, fr in POS_WORDS:
        pairs.append((en, fr, 1, "strong"))
    for en, fr in POS_WORDS_LOOSE:
        pairs.append((en, fr, 1, "loose"))
    for en, fr in POS_PHRASES:
        pairs.append((en, fr, 1, "strong"))
    for en, fr in POS_PHRASES_LOOSE:
        pairs.append((en, fr, 1, "loose"))
    for en, fr in NEG_TRANSLATIONS:
        pairs.append((en, fr, 0, "translation"))
    for en, fr in NEG_TRANSLATION_PHRASES:
        pairs.append((en, fr, 0, "translation"))
    for en, fr in NEG_RANDOM:
        pairs.append((en, fr, 0, "random"))
    return pairs


if __name__ == "__main__":
    ps = all_pairs()
    npos = sum(1 for p in ps if p[2] == 1)
    print(f"{len(ps)} pairs: {npos} positive, {len(ps) - npos} negative")
