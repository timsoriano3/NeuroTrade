"""Check that the documentation still describes the code.

    uv run python scripts/check_docs.py

Documentation drifts because nothing fails when it goes stale. This closes the
three gaps that are mechanically checkable, all of which had actually happened
by the time it was written:

1. a README naming a file that no longer exists, or was moved
2. a documented `make` target that is not in the Makefile
3. a package with code and no README

It deliberately checks only what can be verified without judgement. Whether a
sentence is *true* is not checkable; whether the file it names exists is.

Run by `make docs-check`, which `make lint` includes.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = sorted(
    path
    for path in ROOT.rglob("*.md")
    if not any(part in {".venv", "node_modules", ".git"} for part in path.parts)
    # The local copy of the Notion spec is gitignored and not ours to keep in step.
    and path.name != "TRADER_PLAN.md"
)

SEARCH_ROOTS = ("src", "scripts", "tests", "config", "infra")

CODE_FILE = re.compile(r"`([A-Za-z_][A-Za-z0-9_/.]*\.py)`")
MAKE_TARGET = re.compile(r"(?:^|\$ |&& )make ([a-z][a-z-]*)", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)#]+\.md)\)")

FENCED_BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.DOTALL)
INLINE_CODE = re.compile(r"`([^`\n]+)`")


def _code_only(text: str) -> str:
    """Everything inside fenced blocks and inline code spans, prose discarded.

    Prose says things like "would make the corpus" and "a make target"; matching
    a command pattern against it produces false reports, and a checker that
    reports things that are not problems gets switched off.
    """
    fenced = FENCED_BLOCK.findall(text)
    inline = INLINE_CODE.findall(text)
    return "\n".join([*fenced, *inline])


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_referenced_files() -> list[str]:
    """Every `something.py` in a doc must resolve to a real file.

    Matched by basename, because a README refers to `codec.py` rather than
    spelling out the full path. That is looser than a path check and catches the
    case that actually bites: a file renamed or moved while the prose stayed.
    """
    existing = {path.name for root in SEARCH_ROOTS for path in (ROOT / root).rglob("*.py")}
    problems = []
    for doc in DOCS:
        for name in sorted(set(CODE_FILE.findall(doc.read_text()))):
            basename = Path(name).name
            if basename not in existing:
                problems.append(f"{_relative(doc)}: references {name}, which does not exist")
    return problems


def check_make_targets() -> list[str]:
    """Every documented `make x` must be a real target.

    This is the drift with the sharpest edge: a reader trusts the doc, runs the
    command, and it fails. CLAUDE.md listed six commands that did not exist.
    """
    makefile = (ROOT / "Makefile").read_text()
    targets = set(re.findall(r"^([a-z][a-z-]*):", makefile, re.MULTILINE))

    problems = []
    for doc in DOCS:
        for target in sorted(set(MAKE_TARGET.findall(_code_only(doc.read_text())))):
            if target not in targets:
                problems.append(
                    f"{_relative(doc)}: documents `make {target}`, which is not a target"
                )
    return problems


def check_packages_documented() -> list[str]:
    """Every package holding real code needs a README.

    `src/neurotrade` itself is exempt: the repository README covers the layout,
    and a second one there would be the first to drift.
    """
    problems = []
    for directory in sorted((ROOT / "src" / "neurotrade").rglob("*")):
        if not directory.is_dir() or "__pycache__" in directory.parts:
            continue
        has_code = any(path.name != "__init__.py" for path in directory.glob("*.py"))
        if has_code and not (directory / "README.md").exists():
            problems.append(f"{_relative(directory)}: has code but no README.md")
    return problems


def check_internal_links() -> list[str]:
    """Relative links between documents must resolve."""
    problems = []
    for doc in DOCS:
        for target in sorted(set(MARKDOWN_LINK.findall(doc.read_text()))):
            if not (doc.parent / target).resolve().exists():
                problems.append(f"{_relative(doc)}: broken link to {target}")
    return problems


CHECKS = (
    ("referenced files exist", check_referenced_files),
    ("documented make targets exist", check_make_targets),
    ("packages with code have a README", check_packages_documented),
    ("internal links resolve", check_internal_links),
)


def main() -> int:
    problems: list[str] = []
    for label, check in CHECKS:
        found = check()
        status = f"{len(found)} problem(s)" if found else "ok"
        print(f"  {label:<38} {status}")
        problems.extend(found)

    if problems:
        print()
        for problem in problems:
            print(f"  {problem}")
        print(f"\n{len(problems)} documentation problem(s) across {len(DOCS)} files")
        return 1

    print(f"\n{len(DOCS)} documents checked, no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
