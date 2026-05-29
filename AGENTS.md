# Agent guide — machine-learning-library

This repo is a **curated, machine-readable corpus of ML education**: 923 Markdown
documents (~11M tokens) — research papers, course lecture transcripts, and
canonical explainer articles — each with structured YAML frontmatter. This file
tells an AI agent how to navigate and retrieve from it. (Read by Cursor, Codex,
Copilot, Gemini CLI, Aider, Zed, etc. Claude Code reads `CLAUDE.md`, which points
here.)

## Layout

```
corpus/
├── papers/   391 docs   — arXiv papers (78 full text + 313 abstract+metadata). Filename = arXiv id (1706.03762.md)
├── youtube/  474 docs   — lecture transcripts, grouped by course/channel. Filename = video id
└── web/       58 docs   — explainer articles, grouped by domain
corpus/INDEX.md          — master list of every doc with its title
atlas/                   — human/topic navigation layer (Maps of Content per topic)
atlas/TAGS.md            — the controlled tag vocabulary (authoritative)
```

Files are organized **by source on disk**, but tagged **by topic** in
frontmatter — so retrieve by tag/content, not by folder.

## Frontmatter schema

Every doc starts with a YAML block. Fields you'll use for retrieval:

```yaml
title:    "Attention Is All You Need"      # human title
source:   "arxiv" | "youtube" | "web"
url:      "http://arxiv.org/abs/1706.03762v7"   # ALWAYS cite this
authors:  [...]            # papers
published:"2017-06-12"     # papers (YYYY-MM-DD)
channel:  "Stanford CS25"  # lectures
topics:   [...]            # original free-form tags (papers/web only) — provenance, do not rely on
tags:     [topic/transformers-attention, level/advanced, medium/paper, task/language, technique/attention]
aliases:  ["Attention Is All You Need", ...]
```

The `tags:` field is the controlled, queryable layer. Namespaces:
`topic/` (17 subjects), `level/` (intro·intermediate·advanced·frontier),
`medium/` (paper·lecture·article), `task/` (vision·language·…), `technique/`
(cnn·transformer·diffusion·lora-peft·…). See `atlas/TAGS.md` for the full list.

## Retrieval workflow (do this)

1. **Find candidates.** Prefer full-text search over the corpus — it is the most
   reliable signal (`grep -ril "flash attention" corpus/`, or your tool's search).
   Narrow with tags when useful: `grep -rl "topic/generative-models" corpus/`.
2. **Read the top 3–5 matches** (read the frontmatter + relevant sections, not
   always the whole file — transcripts can be long).
3. **Answer grounded in those docs, and CITE** each claim with the doc's
   frontmatter `url` (and title). Never invent references — every fact here is
   traceable to a real source.
4. If the user asks to learn a topic, use the relevant `atlas/topics/*.md` Map of
   Content as a reading path.

### Caveats
- **Tags are auto-assigned** (keyword rules + a content-reading pass). They're
  good for narrowing, but when a tag filter returns little, **fall back to
  full-text search** — don't conclude the corpus lacks the topic.
- The same lecture can carry multiple `topic/` tags; that's intended.
- 9 Karpathy videos appear under two folders (same content, same tags).
- This repo contains **no original content** — it's a reformatting of others'
  work. Always attribute the original author/source, never this repo.

## Topic vocabulary (the 17 `topic/` tags)

`neural-network-foundations` · `classical-ml` · `computer-vision` ·
`sequence-models-rnn` · `transformers-attention` · `language-models` ·
`efficient-architectures` · `generative-models` · `multimodal` ·
`reinforcement-learning` · `alignment-rlhf` · `reasoning-agents` ·
`efficiency-systems` · `interpretability` · `evaluation-trust` ·
`ml-engineering` · `ai-industry-news`
