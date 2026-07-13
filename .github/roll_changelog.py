#!/usr/bin/env python3
"""Roll CHANGELOG.md's [Unreleased] section into a released version.

Usage: roll_changelog.py <version> <date>

- Moves everything under "## [Unreleased]" beneath a new "## [<version>] - <date>"
  heading and leaves a fresh, empty "## [Unreleased]".
- Prints the rolled version's notes to stdout (for the GitHub release body).
- Exits non-zero if [Unreleased] is missing or empty, so a release never ships
  with empty notes.
"""
import re
import sys
from pathlib import Path


def main() -> None:
    # The notes we print contain Unicode (→, —); force UTF-8 regardless of the
    # host console's default codepage (e.g. cp1252 on Windows).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    if len(sys.argv) != 3:
        sys.exit("usage: roll_changelog.py <version> <date>")
    version, date = sys.argv[1], sys.argv[2]

    path = Path("CHANGELOG.md")
    text = path.read_text(encoding="utf-8")

    # Capture the body between "## [Unreleased]" and the next "## [" (or EOF).
    m = re.search(r"^## \[Unreleased\]\s*?\n(.*?)(?=^## \[|\Z)", text, re.S | re.M)
    if not m:
        sys.exit("CHANGELOG.md: no '## [Unreleased]' section found")
    body = m.group(1).strip("\n")
    if not body.strip():
        sys.exit("CHANGELOG.md: [Unreleased] is empty — add entries before releasing")

    rolled = f"## [Unreleased]\n\n## [{version}] - {date}\n\n{body}\n\n"
    updated = text[: m.start()] + rolled + text[m.end():]
    updated = re.sub(r"\n{3,}", "\n\n", updated)  # tidy blank runs
    path.write_text(updated, encoding="utf-8")

    sys.stdout.write(body + "\n")  # release notes


if __name__ == "__main__":
    main()
