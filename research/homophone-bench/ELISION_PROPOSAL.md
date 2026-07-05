# LLM elision/remining proposal (nvidia/nemotron-3-super-120b-a12b:free)

_In-context advisory from the LLM-in-the-loop; review before encoding into matcher.py._

1. ELIDE_SCHWA_PRE_VOWEL  
   Condition: Word-final schwa /ə/ before a vowel-initial word → delete /ə/  
   Example: EN /də/ → FR /d/ (e.g., "de une" → /dy.n/ → carve as "d'une")  
   Encoding: CHEAP_GAP: ə → ∅ (when followed by vowel)  

2. ELIDE_SCHWA_POST_VOWEL  
   Condition: Word-initial schwa /ə/ after a vowel-final word → delete /ə/  
   Example: EN /lə/ → FR /l/ (e.g., "le une" → /lə.y/ → carve as "l'une")  
   Encoding: CHEAP_GAP: ə → ∅ (when preceded by vowel)  

3. LIAISON_Z_PLUS_VOWEL  
   Condition: Word-final /z/ (from /s/, /x/, /z/) before vowel → retain /z/ as liaison  
   Example: EN /zɪ/ → FR /zi/ (e.g., "les idées" → /le.z‿i.de/ → carve as "les idées")  
   Encoding: Extend LIAISON: s/x/z → z (already present; reinforce as active in vowel context)  

4. LIAISON_T_PLUS_VOWEL  
   Condition: Word-final /t/ or /d/ (from /t/, /d/) before vowel → retain /t/ as liaison  
   Example: EN /tɪ/ → FR /ti/ (e.g., "dit il" → /di.t‿il/ → carve as "dit-il")  
   Encoding: Extend LIAISON: d/t → t (already present; reinforce as active in vowel context)  

5. LIAISON_N_PLUS_VOWEL  
   Condition: Word-final /n/ before vowel → retain /n/ as liaison  
   Example: EN /nɪ/ → FR /ni/ (e.g., "bon ami" → /bɔ.n‿a.mi/ → carve as "bon ami")  
   Encoding: Extend LIAISON: n → n (already present; reinforce as active in vowel context)  

6. H_MUET_LIAISON_BLOCK  
   Condition: Word-initial /h/ (aspirated) blocks liaison/enchaînement from prior consonant  
   Example: EN /zɑʁ/ → FR /zaʁ/ (e.g., "les haricots" → /le a.ʁi.ko/ → *no liaison*, carve as "les haricots")  
   Encoding: Add to LIAISON exceptions: h-aspiré blocks liaison (mark lexically; e.g., "haricot" as [h]₊)  

7. ENCHAINEMENT_VOWEL_VOWEL  
   Condition: Word-final vowel + word-initial vowel → insert /t/ or /z/ glide to avoid hiatus  
   Example: EN /eɪ/ → FR /et/ (e.g., "il a" → /il.a/ → carve as "il t'a" via /t/ insertion)  
   Encoding: Rewrite: V_V → V_tV or V_zV (context-sensitive; e.g., after /i/, /y/, /u/ → /z/; else /t/)  

8. GEMINATE_REDUCTION  
   Condition: Geminate consonant /C:/ → simplify to /C/ in fast speech  
   Example: EN /tː/ → FR /t/ (e.g., "atteint" → /a.tɛ̃/ → carve as "at-teint" simplified)  
   Encoding: CHEAP_GAP: C:C → C (for /t t/, /d d/, /s s/, /n n/, /l l/)  

9. FINAL_CONSONANT_DEVOICING  
   Condition: Word-final voiced obstruent → devoiced in pausa  
   Example: EN /d/ → FR /t/ (e.g., "livré" → /li.vʁe/ → final /ʁ/ not devoiced, but /d/ in "livred" → /li.vʁt/)  
   Encoding: Extend existing devoicing (implied in RHOTIC_MAP context); add: b/d/g → p/t/k word-finally  

10. SCHWA_INSERTION_BREAK_CLUSTER  
   Condition: Impossible French cluster (e.g., /tl/, /dn/) → insert schwa to break  
   Example: EN /dənt/ → FR /də.n.t/ (e.g., "dent" → /dɑ̃/ → but /dənt/ from English → carve as "dent" with schwa epenthesis)  
   Encoding: CHEAP_GAP: ∅ → ə (to break illicit clusters like /dn/, /tl/, /kr/)  

Total: 10 rules. Focus on elision, liaison, hiatus, and repair. All codeable via existing framework extensions.  
Word count: 298.
---
## Closing the loop (iteration 1 review)

- Schwa elision (rules 1–2) is **already encoded** — `ə` is in CHEAP_GAP at 0.18.
- The genuinely-new rules (elision-before-vowel, enchaînement glides, geminate
  reduction, cluster epenthesis, h-aspiré block) are all **context-sensitive**,
  but our matcher scores segment pairs **context-free** (NW over single segments).
- **Loop verdict / next iteration:** to apply connected-speech rules the matcher
  must become context-aware (condition a substitution/gap on the neighbouring
  segment and word boundary). That refactor is the next remine step; the
  context-free wins (voicing-pair EQUIV, schwa gaps) are already in.
