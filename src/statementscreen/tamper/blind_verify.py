"""Rebuild all 200 blind PDFs from the published key and the source batches, and compare.

    uv run python -m statementscreen.tamper.blind_verify --key reports/week7_blind_key.json \\
        --verdicts reports/week7_blind_verdicts.csv --blind-src data/blind_src

For every PDF the key names a source statement, a truth (genuine, or a tamper type), the copy
seed and the donor. This rebuilds each one (a genuine PDF is the source's bytes; a tampered one
is tamper_copy with the recorded seed and donor) and checks that its SHA-256 equals the one in
the key AND the one in the committed verdict file. So the verdicts were made on exactly the PDFs
the key describes, and anyone can repeat the check with only the public files.

It also checks that the key is the one pre-registered (its SHA-256 against the pre-registration
file, read as plain text: no code or git check, so anyone can run this), that every source PDF
and manifest is the one the key hashed, and that the key's PDF-set hash is the rebuilt set's.

The source batches are read from --blind-src (data/blind_src, which stays on disk for this).
They are regenerated from the two source seeds recorded in the key (7101 default dialect, 7102
karan_v1) with the project's generator, whose output was confirmed byte-identical between runs
at step 2; this command checks the sources against the key's hashes rather than regenerating.

Exit 0: every PDF rebuilt byte for byte. Exit 1: something differs (every difference is listed,
none hidden). Exit 2: an input is missing or is not the pre-registered one.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from statementscreen.splits import sha256_file
from statementscreen.tamper import blind_prereg as bpr
from statementscreen.tamper.blind_common import BlindError, pdf_set_sha256, sha256_bytes
from statementscreen.tamper.copy import CopyError, NotApplicable, tamper_copy
from statementscreen.tamper.rebuild import RebuildError

GENUINE = "genuine"
REBUILD_ERRORS = (NotApplicable, CopyError, RebuildError, OSError, ValueError)


def read_inputs(key_path: Path, verdicts: Path, prereg_path: Path) -> tuple[dict, dict]:
    """(key, {name: PDF SHA-256 from the verdict file}), once the key is shown to be the
    pre-registered one."""
    raw = Path(key_path).read_bytes()
    sealed = json.loads(Path(prereg_path).read_text(encoding="utf-8"))["sealed"]
    if sha256_bytes(raw) != sealed["key_sha256"]:
        raise BlindError(f"{key_path} is not the key that was pre-registered (SHA-256 differs)")
    key = json.loads(raw)
    with Path(verdicts).open(newline="", encoding="utf-8") as f:
        committed = {r["name"]: r["pdf_sha256"] for r in csv.DictReader(f)}
    return key, committed


def verify(key: dict, committed: dict[str, str], blind_src: Path) -> dict:
    """Rebuild every PDF; returns {"rebuilt": {name: sha}, "problems": [{name, truth, what}]}."""
    blind_src = Path(blind_src)
    problems: list[dict] = []

    def problem(name: str, truth: str, what: str) -> None:
        problems.append({"name": name, "truth": truth, "what": what})

    for sub, src in sorted(key["sources"].items()):
        manifest = blind_src / src["folder"] / "manifest.csv"
        if not manifest.is_file() or sha256_file(manifest) != src["manifest_sha256"]:
            problem(sub, "-", f"{manifest} is missing or is not the manifest the key hashed")
    if set(committed) != set(key["pdfs"]):
        problem("-", "-", "the verdict file and the key do not name the same PDFs")
    rebuilt: dict[str, str] = {}
    for name, entry in sorted(key["pdfs"].items()):
        truth = entry["truth"]
        folder = blind_src / key["sources"][entry["sub_batch"]]["folder"] / "pdfs"
        source = folder / f"{entry['source']}.pdf"
        try:
            source_bytes = source.read_bytes()
        except OSError as e:
            problem(name, truth, f"source unreadable: {e}")
            continue
        if sha256_bytes(source_bytes) != entry["source_sha256"]:
            problem(name, truth, "the source PDF is not the one the key hashed")
            continue
        try:
            if truth == GENUINE:
                data = source_bytes
            else:
                donor = folder / f"{entry['donor']}.pdf" if entry["donor"] else None
                data, _ = tamper_copy(source, truth, entry["copy_seed"], donor, allow_held_out=True)
        except REBUILD_ERRORS as e:
            problem(name, truth, f"could not rebuild: {type(e).__name__}: {e}")
            continue
        digest = rebuilt[name] = sha256_bytes(data)
        if digest != entry["pdf_sha256"]:
            problem(name, truth, "the rebuilt PDF differs from the key's hash")
        if digest != committed.get(name):
            problem(name, truth, "the rebuilt PDF differs from the committed verdict file's hash")
    if len(rebuilt) == len(key["pdfs"]) and pdf_set_sha256(rebuilt) != key["pdf_set_sha256"]:
        problem("-", "-", "the rebuilt set's hash differs from the key's PDF-set hash")
    return {"rebuilt": rebuilt, "problems": problems}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m statementscreen.tamper.blind_verify",
        description="Rebuild the 200 blind PDFs from the published key and check every hash.",
    )
    p.add_argument("--key", type=Path, default=Path(bpr.PUBLISHED_KEY))
    p.add_argument("--verdicts", type=Path, default=Path(bpr.VERDICTS))
    p.add_argument("--blind-src", type=Path, default=Path("data/blind_src"))
    p.add_argument("--preregistration", type=Path, default=bpr.DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        key, committed = read_inputs(args.key, args.verdicts, args.preregistration)
        result = verify(key, committed, args.blind_src)
    except (BlindError, OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    total, problems = len(key["pdfs"]), result["problems"]
    bad = {p["name"] for p in problems if p["name"] != "-"}
    print(f"Rebuilt {len(result['rebuilt'])} of {total} PDFs; "
          f"{len(result['rebuilt']) - len(bad & set(result['rebuilt']))} identical to the key "
          "and the verdict file")  # fmt: skip
    if problems:
        by_truth = Counter(p["truth"] for p in problems)
        print(f"  {len(problems)} problems, by truth: {dict(sorted(by_truth.items()))}")
        for item in problems:
            print(f"    {item['name']} ({item['truth']}): {item['what']}")
        return 1
    print("  every PDF matches the key and the committed verdicts byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
