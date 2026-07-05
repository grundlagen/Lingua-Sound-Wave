# Chain-web: how it produces homophonic+semantic translation

From chainwebfullS (16,719 S-tier chains). Two edge types:
  ≈ = SOUND hop (homophone, crosses language)   ~ = MEANING hop (same/translated word, stays in-language)

## Mechanism: translation is a PATH, not a pair

  en:root ≈ fr:route ~ en:route ≈ fr:routes      (root -> routes, 3 hops)
  en:bell ≈ fr:belle ~ en:belle ≈ fr:belles
  en:born ≈ fr:borne ~ en:borne ≈ fr:bonnes

The hinge is a PIVOT word that exists in BOTH languages (route, borne, mess,
cross). Path = sound-bridge into FR -> meaning-bridge across the shared pivot ->
sound-bridge to the target. Endpoint is both sound-near AND sense-near the source,
and the chain SHOWS ITS WORK (every hop logged). 30,305 sound hops vs 23,906
meaning hops = balanced alternation.

## Lessons

1. Translation = alternating ≈/~ path. Compose sound (cross-lingual) with meaning
   (in-lingual) to transfer a word to a foreign word carrying both channels.
2. PIVOT cognates are the hinges (route=route, mess=mess) -- the shared-spelling
   words bridge the languages. Build the pivot index first.
3. Multi-hop extends reach (2->6 hops, 5515->1182 chains) -- reaches pairs no
   direct edge connects, quality decays with length. Keep S-tier (quality 1.0).
4. The web is ONE component (earlier census 99.6%): any word reaches any other by
   enough hops -- the giant homophonic-semantic fabric.

## Critical (Fable lens) -- the real gap

The ~ meaning hops here are mostly SURFACE cognate pivots (route~route,
cross~cross), not deep semantics. So chain-web's "semantic" = shared-spelling
bridges, abundant but shallow. True homophonic+SEMANTIC translation (root -> a FR
word meaning root that sounds like a step) needs DENSE meaning edges from MUSE/
multilingual embeddings, not surface cognates. Same gap as the v7 web: sound
fabric is rich + routable; the meaning layer is thin.

## Apply to current work

- Run the chain-router (weave.py) on the v7 web -> bigger chain-web, more pivots.
- Chains give the generation pipeline the meaning-bridge Round Rabbit theming
  lacked (denser than cognate-only meaning edges).
- End goal: swap surface-cognate pivots for embedding meaning edges -> chains that
  route true sense, not just shared spelling. That is the incredible-translation
  unlock.
