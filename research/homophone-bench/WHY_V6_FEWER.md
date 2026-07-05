# Why v6 finds fewer matches than v5 (even with looser settings)

Tested `retrieve_vs_decode.py` — and the premise is half right. The fix is to
combine, not to loosen.

## Per-word matching is NOT the problem

On 50 common English words, v5's **retrieval** and v6's **decoder** are
near-equal: mean combo **0.661 vs 0.660**. So looser gates / finer grain didn't
make v6 worse at finding a single word's match — per word the two methods tie far
more than they differ (retrieve 13 wins, decode 10, **27 ties**).

## But they are DIFFERENT tools, with different misses

- **v5 = retrieval** (`build_dictionary.py`): block the whole French lexicon by
  phoneme-bigram Dice → rank the shortlist with combo → take the **global argmax
  word**. It scans thousands of real candidates.
- **v6 = decoder** (`build_v6.py`): beam-search the phoneme stream into a word
  **sequence**, optimising the beam's path cost. It prunes.

Their errors are independent:
- retrieval wins where the decoder prunes the best whole-word path
  (`all → sol 0.76` vs decode `elle 0.61`; `your → hier 0.72` vs `or 0.60`);
- the decoder wins where retrieval's **blocking or top-N lexicon missed the word**
  (`can → cane 0.73` vs `quand 0.62`; `from → frime 0.77` vs `femme 0.68`).

## So why does v6 have FEWER ENTRIES (2,351 vs v5's 11,788)?

Not the matcher quality — three structural reasons:

1. **Smaller run.** v6 mined the top ~5,083 frequent words; v5 accreted across
   versions over 12,000+ English words **plus** 12k reverse French words.
2. **Best-per-word only.** v6 keeps one French pick per English word; v5 keeps
   **multiple** candidates per word (sea → si/scie/ci/sis…), inflating its count.
3. **No multiword / reverse / pair-bank.** v5 accumulated multiword phrase entries,
   fr→en reverse entries (104), and a 30k historical pair-bank merge. v6 mined only
   en→fr single best.

The decoder pruning (point in §2 above) is a *minor* contributor; the entry-count
gap is mostly run-size + best-per-word + v5's accumulated extras.

## The fix: union the two generators, don't loosen

Looser gates can't help when the *candidate generator* prunes away the best word.
The right v6 build is **retrieval ∪ decoder**, arbiter-ranked:

- **retrieval** for the clean word-for-word entries (v5's strength, global argmax);
- **decoder** for (a) words retrieval's blocking missed, and (b) the multi-word
  filler carves retrieval cannot make at all (line generation).

Run them both, keep the better per word by combo, and v6 ≥ v5 on word entries
*plus* the filler/line material. That is the concrete change to `build_v6.py`:
replace its decoder-only candidate step with retrieve-then-decode, keep-best.

## One-line answer

v6 isn't worse at matching (0.661 ≈ 0.660); it just used a *line-carving* tool to
build a *word* dictionary, on a smaller run with best-per-word and none of v5's
accumulated multiword/reverse/pair-bank entries. Combine retrieval (v5) with the
decoder (v6) and keep both errors' complement.
