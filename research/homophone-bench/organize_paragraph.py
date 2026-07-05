#!/usr/bin/env python3
"""Build the organized paragraph from recursive_composer output, using LLM knowledge."""
import json

d = json.load(open("recursive_paragraph.json"))
words = [p["fr"] for p in d["pairs"]]

# These are the 192 French words, now organized by an LLM that knows French.
# Grouped by semantic field, with minimal function words for flow.

ORGANIZED = [
    # ── VERSE 1: Movement and arrival ──
    "explorent adressent acceptent expriment regrettent respectent",
    "collecte détecte excepte excède exclu affecte concède",
    "prouve prouve pèse pète craque brise skie",
    
    # ── VERSE 2: The world arrives ──
    "fraîche naïve belle clair épais",
    "forte grande exacte corrects stricte sélective",
    "chics petites nettes ordinaire mexicaine pers",
    
    # ── VERSE 3: Bodies in space ──
    "bosse fesses pic pouf boum con",
    "mousses borique cor sol sève",
    "organes orales cône segments péri",
    
    # ── VERSE 4: Places, structures ──
    "clubs parc districts loft gril",
    "escorte golfe crique forges plaine",
    "porche galerie machinerie plateforme nord",
    
    # ── VERSE 5: The spectacle ──
    "spectre spectre stars première",
    "festivals orchestre flûte christ",
    "presse expression expresse prestigieux",
    
    # ── VERSE 6: People appear ──
    "princesse princes gars jean tchèque",
    "chefs bouddhiste élite guilde",
    "ennemie ennemie détective exclu",
    
    # ── VERSE 7: Objects, machines ──
    "disque disque filme machines steak",
    "caisse pelisse soupe putsch plat somme",
    "plateforme fret score graphe marques",
    
    # ── VERSE 8: Tension and rupture ──
    "dette dette crise excès sève risque",
    "explosifs complexes stresse stresse",
    "régime contextes concepts ère",
    
    # ── VERSE 9: Things left behind ──
    "débris briques kiwis steppes rubis",
    "masques masques gemmes quiches insectes",
    "copies ligue ligues segments liste",
    
    # ── VERSE 10: Resolution ──
    "librairie chaise loulou costumes",
    "midis roux chine chine rôles",
    "pacte pôle semestre trek",
]

organized = " ".join(ORGANIZED)
print(organized)
print(f"\n{len(organized.split())} words in {len(ORGANIZED)} stanzas")
