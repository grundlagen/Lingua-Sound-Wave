"""EN phoneme equivalence map + linguistic rules, integrated from the user's Drive phoneme_mapping_reference.py (their own work). A richer hand-curated EQUIV than matcher.EQUIV; the tournament tests it vs the gold-learned map."""

phoneme_mapping = {
    # Vowel Phonemes
    'i': ['i', 'ɪ', 'iː'],                          # High front vowels, long/short interchange
    'ɪ': ['i', 'ɪ'],                                 # Close front vowel, similar in many accents
    'eɪ': ['e', 'eɪ', 'ɛ', 'aɪ'],                    # Diphthong mapping to monophthongs or other diphthongs
    'ɛ': ['ɛ', 'e', 'æ', 'aɪ'],                      # Mid front, flexible in accents (e.g., Northern English "e" vs "æ")
    'æ': ['a', 'æ', 'ɑ'],                             # Low front, often replaced by "a" or "ɑ" in regional accents
    'ɑ': ['a', 'ɑ', 'æ', 'ɔ'],                        # Open back, overlaps with "a" and rounded in some dialects
    'ɔ': ['o', 'ɔ', 'ɒ'],                             # Mid back, can interchange with "o" or low back rounded vowel
    'ə': ['ə', 'ʌ', 'ɜ', 'ɨ'],                        # Schwa, interchangeable in unstressed positions
    'ʌ': ['ə', 'ʌ', 'ɑ'],                             # Open-mid back, often overlaps with schwa
    'u': ['u', 'ʊ'],                                  # High back, interchanges with rounded versions
    'ʊ': ['u', 'ʊ', 'oʊ'],                            # Close back rounded, also maps with certain diphthongs
    'ɜ': ['ɜ', 'ə', 'ɝ'],                             # Mid central vowel, similar to schwa or rhotic version
    'ɒ': ['ɑ', 'ɔ', 'ɒ', 'æ'],                        # Back rounded, can vary regionally to open front

    # Consonant Phonemes
    't': ['t', 'ʔ', 'ɾ', 'tw'],                       # Alveolar stop, changes to glottal stop, flap, or labialized variant
    'd': ['t', 'ɾ', 'd'],                             # Alveolar stop, interchange with voiced or voiceless versions
    's': ['s', 'z', 'θ'],                             # Alveolar fricative, voicing or assimilation to dental
    'z': ['s', 'z', 'ð'],                             # Voiced fricative, voiceless equivalent or dental substitution
    'ʃ': ['ʃ', 'ʒ', 'tʃ'],                            # Postalveolar fricative, can become affricate or voiced version
    'ʒ': ['ʃ', 'ʒ', 'dʒ'],                            # Voiced version, matches affricate and fricative pairs
    'n': ['n', 'ŋ', 'ɱ', 'ʔ'],                        # Nasals flexible depending on position or assimilation
    'ŋ': ['n', 'ŋ', 'ɲ'],                             # Velar nasal, interchangeable in word-final or medial
    'l': ['l', 'ɫ', 'ʟ', 'ʊ'],                        # Lateral approximants, dark "l" (syllable-final)
    'r': ['r', 'ɹ', 'ɻ', 'ə', 'ɾ', ''],               # Rhotics, dropped or replaced with schwa in some dialects
    'p': ['p', 'b', 'pʰ', 'ʔ'],                       # Voiceless bilabial stop, can become glottal or aspirated
    'b': ['p', 'b'],                                  # Voiced bilabial, softens or voices to pair
    'k': ['k', 'g', 'kʰ', 'ʔ'],                       # Velar stop, can become glottal or aspirated
    'g': ['k', 'g'],                                  # Voiced velar, interchangeable with unvoiced equivalent
    'h': ['', 'h', 'ʔ'],                              # H-dropping and glottal substitution
    'f': ['f', 'v', 'θ'],                             # Fricative voicing lenition or substitution with dental
    'θ': ['θ', 'f', 't'],                             # Th-fronting or substitution with "t" in certain accents
    'ð': ['ð', 'v', 'd'],                             # Voiced equivalent, voicing changes or replaced by stop
    'j': ['', 'j', 'tʃ', 'dʒ'],                       # Yod-coalescence, yod-dropping, or affricate formation
    'w': ['w', 'ʍ'],                                  # Labialized velar or voiceless version (wh-sound)
    'v': ['v', 'f', 'w'],                             # Labio-dental voiced to voiceless substitution
    'm': ['m', 'n'],                                  # Labial to alveolar nasal in connected speech

    # Glottal Stops and Gap-Filling
    'ʔ': ['ʔ', ''],                                   # Can be a gap or silent in blending two words
    'ɾ': ['t', 'd', 'ɾ'],                             # Flap, typical in American English (e.g., butter)
    'ʊ': ['ʊ', 'ɫ'],                                  # Dark l as final sound alternative (syllabic l)

    # Aspiration and Lenition
    'pʰ': ['p', 'pʰ'],                                # Aspirated or unaspirated based on context
    'tʰ': ['t', 'tʰ'],                                # Alveolar, aspirated/unaspirated
    'kʰ': ['k', 'kʰ'],                                # Velar, aspirated/unaspirated

    # Reduction and Intrusive Elements
    'aɪ': ['aɪ', 'eɪ', 'ɪ', 'aː'],                    # Diphthong reduced to monophthong in some accents
    'ɔɪ': ['ɔɪ', 'ɔ', 'o'],                           # Diphthong reduction or smoothening
    'aʊ': ['aʊ', 'æ', 'o'],                           # Dipthong reduced based on accent or emphasis
    'eɪ': ['eɪ', 'e', 'ɛ'],                           # Reduction or merging with other vowels
    'eə': ['eə', 'ɜ', 'ə'],                           # Reduction to schwa or smoothened articulation
    'ʌɪ': ['aɪ', 'aː'],                               # Similar in diphthongs, common in dialect shift

    # Reduction to Schwa and Stress Variability
    'ə': ['ə', 'ɪ', 'a', 'ɜ'],                        # Schwa replaced with context-specific vowels
    'ɜ': ['ɜ', 'ə', 'ɜr'],                            # Stressed/unstressed variations based on regional stress

    # Lateralization and Coarticulation
    'tʃ': ['tʃ', 'dʒ', 'ʃ'],                          # Affricate reduced to fricative or voiced counterpart
    'dʒ': ['tʃ', 'dʒ', 'ʒ'],                          # Affricate voiced/voiceless interplay
    'tw': ['t', 'tw'],                                # Labialization and added rounding (e.g., "twenty")
}

### Additional Linguistic Features & Considerations

rules = {
    "aspiration_lenition_fortition": lambda p1, surrounding: p1 in ['p', 't', 'k'] and surrounding in ['pʰ', 'tʰ', 'kʰ'],
    "glottal_stops_for_gaps": lambda p1, p2: (p1 == '' or p2 == 'ʔ'),
    "intrusive_linking_r": lambda p1, p2: p1 == 'r' or p2 == 'r',
    "th_fronting": lambda p1, p2: p1 == 'θ' and p2 in ['f', 't'],
    "dark_l_vocalized_l": lambda p1, position: p1 == 'l' and position == 'syllable-final',
    "vowel_reduction_schwa": lambda p1, position: p1 in ['a', 'e', 'o', 'u'] and position == 'unstressed',
    "nasal_place_assimilation": lambda p1, next_p: p1 == 'n' and next_p in ['p', 'b', 'm'],
    "flapping": lambda p1, p2: p1 in ['t', 'd'] and p2 == 'ɾ',
    "coarticulation_effects": lambda p1, context: p1 in ['h'] and context == 'casual',
    "h_dropping": lambda p1, position: p1 == 'h' and position == 'unstressed',
    "nasal_g_dropping": lambda p1, position: p1 == 'ŋ' and position == 'word-final',
   "affricate_fricative_coalescence": lambda p1, p2: (p1 == 'tʃ' and p2 == 'ʃ') or (p1 == 'dʒ' and p2 == 'ʒ'),
    "prosodic_markers_stress_patterns": lambda stress: stress in ['primary', 'secondary'],  # Recognizing stress levels
    "glottal_reinforcement": lambda p1, stress: p1 == 'ʔ' and stress == 'stressed',
    "meter_tempo_accentuation": lambda context: context in ['poetry', 'lyrical'],  # Applying rhythmic modifications
    "linking_r": lambda p1, p2: p1 == 'r' or p2 == 'r',  # Linking R in non-rhotic accents
    "vowel_harmony": lambda p1, p2: p1 in ['i', 'e'] and p2 in ['i', 'e'],  # Vowel harmonization
     "devoicing_final_consonants": lambda p1, position: p1 in ['b', 'd', 'g'] and position == 'final',  # Devoicing at end of word
}
