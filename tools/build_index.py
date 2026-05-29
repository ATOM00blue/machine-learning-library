#!/usr/bin/env python3
"""Regenerate corpus/INDEX.md as clickable, forward-slash Markdown links.

The original INDEX used Windows backslash paths that don't resolve on GitHub or
in Obsidian. This rebuilds it from the live frontmatter: grouped by source, with
each entry a real relative link rendered as the document's title.
"""
from __future__ import annotations
import glob, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")


def title_of(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n", 40)
    for line in lines[1:]:
        m = re.match(r'^title:\s*"?(.*?)"?\s*$', line)
        if m:
            return m.group(1)
        if line.strip() == "---":
            break
    return os.path.basename(path)


def main():
    by_group = defaultdict(list)
    n = 0
    for path in glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True):
        if os.path.basename(path) == "INDEX.md":
            continue
        rel = os.path.relpath(path, CORPUS).replace(os.sep, "/")
        cat = rel.split("/")[0]                      # papers / youtube / web
        group = rel.split("/")[1] if "/" in rel.split("/", 1)[1] else cat
        by_group[(cat, group)].append((title_of(path), rel))
        n += 1

    out = ["# Corpus Index", "",
           f"_Machine-generated — {n} documents. Every link is relative to this file._",
           ""]
    headings = {"papers": "## Papers", "youtube": "## Lectures (YouTube)", "web": "## Web articles"}
    for cat in ("papers", "youtube", "web"):
        out.append(headings[cat])
        out.append("")
        groups = sorted(g for (c, g) in by_group if c == cat)
        for group in groups:
            items = sorted(by_group[(cat, group)], key=lambda x: x[0].lower())
            if cat != "papers":
                out.append(f"### {group}  ({len(items)})")
            for title, rel in items:
                safe = title.replace("[", "(").replace("]", ")")
                out.append(f"- [{safe}]({rel})")
            out.append("")
    open(os.path.join(CORPUS, "INDEX.md"), "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote corpus/INDEX.md with {n} entries (forward-slash links)")


if __name__ == "__main__":
    main()
