# tools/

Scripts that build the topic / Obsidian / agent layer **on top of** the corpus.
They are **additive and idempotent** — they only append `tags:`/`aliases:` to
frontmatter and generate `atlas/`; they never move, rename, or delete corpus
files, and never touch the original `topics:` field. Re-running reproduces the
same output.

## Pipeline (run in order)

```bash
python tools/clean_corpus.py    # 1. normalize line endings to LF, strip PDF-extraction control junk
python tools/tag_corpus.py      # 2. append controlled `tags:` + `aliases:` to every doc
python tools/build_index.py     # 3. regenerate corpus/INDEX.md (clickable, forward-slash links)
python tools/build_atlas.py     # 4. generate the atlas/ navigation layer from the tags
```

No third-party Python deps for the build (stdlib only). `examples/rag_quickstart.py`
is separate and lists its own deps.

## How tagging works (`tag_corpus.py`)

Each doc gets a controlled `tags:` list across five namespaces — `topic/`,
`level/`, `medium/`, `task/`, `technique/` (see [`../atlas/TAGS.md`](../atlas/TAGS.md)).
Sources of truth, in order:

1. **Existing `topics:`** (papers/web) → mapped to controlled topics.
2. **Course priors** for focused courses (CS231n→vision, CS229→classical, …).
3. **Title keywords** for everything else.
4. **`overrides.json`** — the highest-confidence source: topic tags produced by an
   agent that *read the actual transcript* for the 203 docs that title-keywords
   couldn't classify well (all of yannic + deeplearningai, plus survey-course
   fallbacks). This file is committed and is the curated correction layer; edit it
   to fix any tag and re-run `tag_corpus.py`.

`medium/` is deterministic from the folder; `level/` from course/date; `task/`
derived from topics. Multi-topic docs are expected and intended.

## Derived/scratch files (git-ignored)

`.tags_manifest.json`, `.ambiguous.json` are regenerated each run. `overrides.json`
is **not** ignored — it's curated data.
