# CLAUDE.md

This repository is a curated ML education corpus (923 Markdown docs: arXiv
papers, lecture transcripts, explainer articles), each with structured YAML
frontmatter.

**The full agent guide is in [`AGENTS.md`](AGENTS.md) — read it before retrieving.**
For an interactive, on-demand retrieval procedure, the project skill at
`.claude/skills/ml-library/SKILL.md` is also available.

## Quick retrieval protocol

1. Full-text search the corpus first (`grep -ril "<query>" corpus/`); narrow with
   tags (`grep -rl "topic/<slug>" corpus/`) when helpful.
2. Read the top 3–5 matches (frontmatter + relevant sections).
3. Answer grounded in them and **cite each claim with the doc's frontmatter
   `url`** — never fabricate references.
4. If a tag filter returns little, fall back to full-text search before
   concluding the corpus lacks a topic. Tags are auto-assigned.

The 17 `topic/` tags and the full `tags:` schema are in `atlas/TAGS.md`.
This repo contains no original content — always attribute the original source.
