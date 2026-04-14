"""Interactive training mode for attorneys learning the tool.

Walks through six bundled sample drafts, each demonstrating a different
hallucination class (clean baseline, fabricated citations, fake quote,
short-form swap, wrong year, multi-kind extraction). For each draft the
attorney sees:

  * the draft text,
  * an explanation of what to look for,
  * the actual report the tool produced.

Run with:
    hd-train
    hd-train --offline       # skip outbound API calls
    hd-train --no-pause      # don't wait for keypress between drafts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends.courtlistener import CourtListenerClient
from .backends.multi import MultiJurisdictionValidator
from .backends.restatement import RestatementCorpus
from .detector import HallucinationDetector
from .policy import FirmPolicy
from .providers import StubProvider


_DRAFTS_DIR = Path(__file__).with_name("training_drafts")


def _load_manifest() -> list[dict]:
    with (_DRAFTS_DIR / "manifest.json").open() as f:
        return json.load(f)["drafts"]


def _build_detector(offline: bool) -> HallucinationDetector:
    backends: list = [RestatementCorpus()]
    opinion_fetcher = None
    if not offline:
        try:
            cl = CourtListenerClient()
            backends.insert(0, cl)
            opinion_fetcher = cl.fetch_opinion_text
        except ImportError:
            pass
    return HallucinationDetector(
        creator=StubProvider(lambda p, s: "", model="unused"),
        validator=MultiJurisdictionValidator(backends=backends),
        opinion_fetcher=opinion_fetcher,
        policy=FirmPolicy(),
        score_uncertainty=False,
    )


def _print_section(title: str) -> None:
    bar = "=" * 78
    print()
    print(bar)
    print(f"  {title}")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hd-train",
        description="Walk through bundled training drafts.",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Skip outbound API calls; demonstrates pure local mode.",
    )
    parser.add_argument(
        "--no-pause", action="store_true",
        help="Don't pause between drafts.",
    )
    parser.add_argument(
        "--draft", type=int, metavar="N",
        help="Run only draft N (1-indexed).",
    )
    args = parser.parse_args(argv)

    detector = _build_detector(args.offline)
    drafts = _load_manifest()

    selection = (
        [drafts[args.draft - 1]] if args.draft else drafts
    )

    print(
        "Hallucination Detector - Training Mode\n"
        f"  {len(selection)} draft(s) | offline={args.offline}"
    )
    if not args.offline:
        print(
            "  (API calls go to public legal databases only; no client data leaves "
            "this machine.)"
        )

    for i, entry in enumerate(selection, start=1):
        path = _DRAFTS_DIR / entry["file"]
        text = path.read_text()

        _print_section(f"Draft {i}/{len(selection)}: {entry['title']}")
        print()
        print("DRAFT:")
        for line in text.rstrip().splitlines():
            print(f"  | {line}")
        print()
        print("WHAT TO LOOK FOR:")
        for line in _wrap(entry["what_to_look_for"], width=72):
            print(f"  {line}")

        report = detector.check_draft(text, use_api=not args.offline)
        print()
        print("DETECTOR REPORT:")
        print(_indent(report.summary(), "  "))
        if report.citations:
            print()
            print("  Citations:")
            for c in report.citations:
                print(f"    [{c.status.value:>16}] {c.kind.value:<13} {c.normalized}")

        if not args.no_pause and i < len(selection):
            try:
                input("\n  [Enter to continue, Ctrl-C to stop] ")
            except (KeyboardInterrupt, EOFError):
                print()
                return 0

    print()
    print("Training complete. You can now run `hd path/to/your-draft.txt`.")
    return 0


def _wrap(text: str, width: int = 72) -> list[str]:
    import textwrap
    return textwrap.wrap(text.strip(), width=width)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
