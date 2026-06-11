"""Small smoke test for the composition-ready v5 release."""
from __future__ import annotations

import json
from pathlib import Path


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    entries = json.load(open("dictionary-v5.json", encoding="utf-8"))
    generated = [e for e in entries if e.get("source_stage") == "v5.2_generated_validated"]
    assert_true(len(entries) >= 10846, "dictionary shrank unexpectedly")
    assert_true(len(generated) >= 900, "generated matches were not merged")
    for en, fr in [("wrest", "reste"), ("truce", "trousse"), ("tress", "tresses")]:
        match = [e for e in entries if e.get("en") == en and e.get("fr") == fr]
        assert_true(match, f"missing generated example {en}~{fr}")
        assert_true(bool(match[0].get("fr_ipa")), f"missing fr_ipa for {en}~{fr}")
        assert_true(bool(match[0].get("align")), f"missing alignment for {en}~{fr}")
    the = [e for e in entries if e.get("en") == "the" and e.get("fr") == "de"]
    assert_true(the and the[0].get("composition_only"), "missing composition-only the->de glue")
    assert_true(the[0].get("usable_for_composition"), "the->de glue is not usable for composition")

    for path in [
        "dictionary-v5.tsv",
        "composition-index.json",
        "composition-lots.json",
        "composition-lines.json",
        "mapping-web.json",
        "mapping-walks.tsv",
        "muse-status.json",
    ]:
        assert_true(Path(path).exists() and Path(path).stat().st_size > 0, f"missing output {path}")

    lines = json.load(open("composition-lines.json", encoding="utf-8"))["lines"]
    assert_true(any(row["usable_for_composition"] for row in lines), "no composed line passed QC")
    graph = json.load(open("mapping-web.json", encoding="utf-8"))
    assert_true(graph["counts"]["sound_edges"] > 10000, "sound graph is too small")
    assert_true(graph["counts"]["fragment_edges"] >= 2600, "fragment graph is too small")
    assert_true(graph["counts"]["meaning_edges"] > 0, "meaning graph has no safe semantic edges")
    print("smoke ok")
    print(f"entries={len(entries)} generated={len(generated)} walks={graph['counts']['walks']}")


if __name__ == "__main__":
    main()
