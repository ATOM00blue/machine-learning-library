---
name: ml-library
description: Retrieve and cite from this machine-learning education corpus (923 docs — arXiv papers, course lectures, explainer articles). Use when the user asks an ML/deep-learning question, wants to learn a topic, asks for sources/papers/lectures on a concept, or wants a study path grounded in this repo.
---

# Retrieving from the ML library corpus

This repo holds 923 Markdown docs under `corpus/` (papers/, youtube/, web/), each
with YAML frontmatter (`title`, `url`, `tags`, …). Your job: answer ML questions
**grounded in these documents, with citations** — never from memory alone.

## Procedure

1. **Search broadly first.** Full-text search is the most reliable signal:
   - `grep -ril "<phrase>" corpus/` for concepts (e.g. `"rotary position"`).
   - Narrow by tag when useful: `grep -rl "topic/<slug>" corpus/` (slugs in `atlas/TAGS.md`).
   - For a learning path, open `atlas/topics/<Topic>.md` (a curated Map of Content).
2. **Read the top 3–5 hits.** Read each doc's frontmatter plus the relevant
   sections (transcripts are long — don't always read the whole file).
3. **Synthesize and cite.** Every claim → the source doc's frontmatter `url` and
   `title`. If sources disagree, say so. Never invent a citation.
4. **If tags miss, fall back to full-text search** before saying the corpus lacks
   something — `tags:` are auto-assigned and approximate.

## The 17 topics

`neural-network-foundations` `classical-ml` `computer-vision`
`sequence-models-rnn` `transformers-attention` `language-models`
`efficient-architectures` `generative-models` `multimodal`
`reinforcement-learning` `alignment-rlhf` `reasoning-agents`
`efficiency-systems` `interpretability` `evaluation-trust`
`ml-engineering` `ai-industry-news`

## Rules

- This repo is a curation of others' work — **attribute the original author**, not
  this repo.
- Prefer primary sources (papers) for technical claims; lectures/articles for
  intuition and explanation.
- When the user wants depth, assemble multiple sources on the topic and trace each
  point back to its origin (see `examples/self-attention-study-note.md` for the
  target output style).
