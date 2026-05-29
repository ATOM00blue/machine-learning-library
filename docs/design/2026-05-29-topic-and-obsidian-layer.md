# Design: Topic Navigation + Obsidian + Agent Layer for `machine-learning-library`

> **Status:** approved design, ready for implementation planning.
> **Date:** 2026-05-29
> **Scope:** Turn a source-organized corpus of 590 docs (~10M tokens) into a topic-navigable, agent-readable knowledge base — **without moving, renaming, or deleting a single file.**

---

## 1. Context & motivation

`machine-learning-library` is a hand-curated corpus of ML education (78 arXiv papers, 474 lecture transcripts, 38 web articles) normalized to Markdown with full provenance. It was published 2026-05-28 and took **94 stars in its first 2 days**, so inbound links, existing clones, and current RAG pipelines must keep working unchanged.

Today the repo is organized **by source**:

```
corpus/
├── papers/    78 docs   — named by arXiv id (1409.0473.md)
├── youtube/  474 docs   — grouped by channel/course, named by video id
└── web/       38 docs   — grouped by domain (jalammargithubio/, distillpub/…)
```

The owner wants it arranged **topic-wise**, "easy for anyone to use," with **Obsidian support** (drop the repo into an Obsidian vault) and the ability to **connect an agent** to the corpus.

### Verified facts (ground truth, confirmed against the live corpus)

- 590 docs total: 78 papers, 474 youtube, 38 web.
- Every doc has YAML frontmatter: `title, source, url, authors, published` and source-specific fields (`arxiv_id`, `video_id`, `channel`, `uploader`, `duration_sec`, `categories`, `fetched_at`, …).
- **Topic coverage is uneven:** **91 docs** have a non-empty `topics:` field (53 papers + 38 web). **All 474 YouTube docs have `topics: []`** (empty). So "arrange topic-wise" is primarily a *tagging project*, not a re-shuffle.
- Existing `topics:` vocabulary is free-form and inconsistent (`nn`, `transformer`, `attention`, `cnn`, `efficient`, `frontier`, …) — no controlled vocabulary.
- **Encoding issue:** ~all paper files use CRLF line endings, which breaks line-anchored tooling (grep `^topics:`, some YAML parsers) and masked topic tags during analysis.
- 9 Karpathy videos are duplicated across two folders (`andrej-karpathy-channel` and `andrej-karpathy-neural-networks-zero-to-`): `kCc8FmEb1nY, l8pRSuU81PU, P6sfmUTpUmc, PaCmpygFfXo, q8SA3rM6ckI, t3YJ5hKiMQ0, TCH_1BHY58I, VMj-3S1tku0, zduSFxRajkE`.

---

## 2. Goals & principles

Serve **both** human browsing (Obsidian) **and** agent/RAG consumption from **one** structure.

1. **Additive only.** Add YAML *fields*, new *files*, and new top-level *folders*. Never move, rename, or remove. Deleting the new layer returns the repo to today's exact state.
2. **One structure, two audiences.** The same `tags:` + Maps-of-Content (MOC) layer powers human browsing and agent/RAG retrieval. No forked structures.
3. **Graceful degradation across four tiers, each strictly better, none required:**
   - github.com (plain Markdown) → curated wikilinks render as clickable titled lists.
   - vanilla Obsidian (no plugins) → aliases, tags, curated links, graph edges all work.
   - Obsidian + bundled plugins → live Dataview tables + readable graph nodes.
   - grep / RAG → `tags:`, `aliases:`, `title:`, `topics:` all greppable.
4. **Grounded vocabulary.** Every topic tag maps to terms that actually appear in the corpus. Nothing invented.
5. **Provenance preserved.** The existing free-form `topics:` field is never modified; the controlled `tags:` field is parallel and new, so every current consumer keeps working byte-for-byte.

---

## 3. Topic taxonomy (final controlled vocabulary)

**Primary facet = `topic`, 17 tags, multi-tag allowed.** Counts are de-duplicated per-document estimates (existing free-form tags are exact; YouTube magnitudes come from validated course + title-keyword passes). ~15–20% of borderline docs may shift between adjacent topics on full-text inspection.

| Topic slug | ~Docs | What it covers |
|---|---:|---|
| `neural-network-foundations` | 46 | MLPs, forward pass, backprop, activations + optimization/regularization (SGD/Adam, batch/layer norm, dropout, init, LR schedules, training-debugging recipes). 3b1b ch1–4, Karpathy micrograd/makemore, CS231n/CS229/CS224n intros + training lectures. |
| `classical-ml` | 16 | Non-neural methods: linear/logistic regression, SVM/kernels, trees/ensembles, naive Bayes/GDA, EM, PCA/ICA, learning theory, cross-validation. Almost entirely Stanford CS229. |
| `computer-vision` | 30 | CNNs (ResNet, VGG, Inception, DenseNet), classification, detection (R-CNN, YOLO), segmentation, vision transformers (ViT, DETR, MAE). CS231n, MIT 6.S191 CNN editions, d2l ch7. |
| `sequence-models-rnn` | 26 | Pre-transformer sequence/NLP: word embeddings (word2vec/GloVe), tokenization, RNN/LSTM/GRU, seq2seq + Bahdanau attention, NMT, parsing, coref, QA. CS224n, MIT, Karpathy WaveNet/tokenizer. |
| `transformers-attention` | 67 | The transformer architecture and attention as a building block: self-/multi-head attention, encoder-decoder, positional encodings (RoPE/ALiBi), induction heads, architecture explainers. All of CS25, CS224n, 3b1b ch5–6. Largest cluster. |
| `language-models` | 50 | Building LLMs end-to-end: pretraining objectives/data, BERT/GPT/T5/LLaMA/Mistral/DeepSeek reports, scaling laws, emergent abilities, MoE as a modeling choice. CS336 "from scratch", Karpathy GPT builds. |
| `efficient-architectures` | 18 | Sub-quadratic/long-context model designs: sparse/linear attention (Reformer, Linformer, Performer, Longformer, FlashAttention), long-context (Infini-attention, attention sinks), SSM/linear-recurrent (Mamba, RWKV, xLSTM, RetNet). |
| `generative-models` | 62 | One hub for deep generative modeling: VAEs/autoencoders, GANs, flows, energy/score-based, autoregressive, and diffusion incl. latent/stable + text-to-image/video. CS236, fast.ai SD course, MIT generative editions. |
| `multimodal` | 12 | Cross-modal models: CLIP, text-conditional image gen (DALL-E/unCLIP), audio/speech transformers, general multimodal lectures. CS25 audio/vision/biomedical, MIT ASR, V-JEPA. |
| `reinforcement-learning` | 17 | RL as a subfield independent of LLM post-training: MDPs, value/policy iteration, Q-learning, policy gradients, deep RL, imitation, decision transformers, game agents (AlphaGo/Tensor/Geometry), robotics control. |
| `alignment-rlhf` | 16 | LLM post-training to preferences: SFT/instruction tuning, RLHF/InstructGPT, reward modeling, DPO/ORPO/GRPO, constitutional/safety alignment, red-teaming. CS336 alignment, CS224n post-training. |
| `reasoning-agents` | 40 | Eliciting reasoning + tool-using agents: chain-of-thought, self-consistency, tree-of-thoughts, test-time compute, reasoning models (o1/R1), ReAct/Toolformer, **RAG/retrieval**, agentic planning. |
| `efficiency-systems` | 22 | Systems-level cheap adapt/train/serve: quantization (GPTQ/AWQ/int8/QLoRA), PEFT (LoRA/DoRA), CUDA/Triton kernels, parallelism, KV-cache/paged-attention serving (vLLM), speculative decoding. CS336 systems lectures. |
| `interpretability` | 9 | Reverse-engineering trained models: mechanistic interp (circuits, superposition, induction heads), feature viz, probing, knowledge editing (ROME). transformer-circuits, distill, CS25/CS224n interp. |
| `evaluation-trust` | 14 | Measuring/stress-testing: benchmarks (MMLU/BIG-Bench/HumanEval), eval methodology, uncertainty/calibration, robustness/adversarial, bias/fairness, privacy, hallucination. CS336/CS236/CS224n eval lectures. |
| `ml-engineering` | 60 | Hands-on build/ship with no scientific-subfield home: production agent engineering, app/agent frameworks & ops, dev tooling (FastHTML, nbdev, Mojo, CUDA-for-Python, APL), prompt engineering. **Honest catch-all** so it doesn't pollute `reasoning-agents`. |
| `ai-industry-news` | 45 | News/commentary/society/meta: model-release roundups, company drama, policy/regulation, AGI debate, conference coverage, applied "AI for X" surveys, channel misc. **Escape valve** for yannic ML-News, NeurIPS coverage, MIT applied/ethics guests. |

### Reconciliation notes (why these merges/splits)

- Generative family collapsed to ONE hub (GAN ≈2, VAE ≈4 vs diffusion ≈38 title hits — splitting starves the small ones); specific architecture carried in the secondary `technique` facet.
- MoE folded into `language-models`; sparse/linear attention + SSM merged into one `efficient-architectures` hub.
- Kept `transformers-attention` vs `language-models` split (owner already maintains distinct `transformer` and `llm` tags; a doc can carry both — architecture-explainer → transformers, whole-model pretraining/scaling → language-models).
- Foundations + optimization/training fused; embeddings/tokenization folded into `sequence-models-rnn`; RAG folded into `reasoning-agents`.
- Graph/GNN (only 2–3 docs) too small for a primary hub → demoted to a secondary `task-domain` value.

### Secondary facets (cross-cutting, queryable independently)

| Facet | Values |
|---|---|
| `level` | `intro` · `intermediate` · `advanced` · `frontier` |
| `medium` | `paper` · `lecture` · `article` (deterministic from `corpus/papers\|youtube\|web`) |
| `task-domain` | `vision` · `language` · `speech-audio` · `multimodal` · `graph` · `rl-control` · `tabular-classical` · `general` |
| `technique` | `mlp` `cnn` `rnn-lstm` `transformer` `attention` `diffusion` `gan` `vae` `normalizing-flow` `ssm` `moe` `lora-peft` `quantization` `rlhf` `dpo` `ppo` `cot` `rag` `flashattention` `embeddings` |

**Decision (resolved):** the existing free-form `frontier` value is a recency/level signal, not a subject → it is **reclassified to the `level/frontier` facet and dropped from the topic vocabulary**. Topics stay strictly subject-based.

`level` assignment = course pedagogical default + title-keyword/date override: intro = 3b1b / MIT-intro / karpathy-zero-to-hero; intermediate = CS229/230/231n/224n/fastai; advanced = CS236/336/25; frontier = recent SOTA papers (≈2023+) + yannic paper-explainers + AI-Dev talks. Older canonical papers take `advanced` (or their subject-appropriate level), not `frontier` — `frontier` means "current research edge," not "is a paper."

Facet tag prefixes: `topic/`, `level/`, `medium/`, `task/` (for task-domain values), `technique/`.

---

## 4. Frontmatter changes

**Nothing removed.** Every existing field stays exactly as-is, including the existing `topics:` field (untouched provenance of the original hand-tagging).

**Fields ADDED:**

| New field | Type | Purpose |
|---|---|---|
| `tags` | list, namespaced | The controlled, Obsidian-native nav layer: `topic/<slug>` + secondary facets `level/*`, `medium/*`, `task/*`, `technique/*`. Powers the tag pane, graph filtering, and `LIST FROM #tag`. Parallel to `topics:` so zero risk to existing consumers. |
| `aliases` | list | Obsidian wikilink targets, derived mechanically from the existing `title:`. Makes `[[Attention Is All You Need]]` resolve against opaque filenames like `1706.03762.md`. Highest-value human-browsing addition. |

`tags:` uses `prefix/value` nesting so the axes (topic / level / medium / task / technique) don't collide and Obsidian groups them in a collapsible tag pane. `level` and `medium` are single-valued; primary `topic` tags may be multi-valued.

**Example — `corpus/papers/2001.04451.md` (Reformer), additive only:**

```yaml
# ...all existing fields unchanged...
topics: ["attention", "efficient"]                 # UNCHANGED
tags: [topic/efficient-architectures, topic/transformers-attention,
       technique/flashattention, level/advanced, medium/paper, task/language]
aliases: ["Reformer", "Reformer: The Efficient Transformer"]
```

### YouTube backfill plan (all 474 docs have `topics: []`)

**Strategy:** course-prior base tags (≈100% coverage) + title-keyword overrides (per-video sharpening). The 9 duplicate Karpathy videos must be tagged identically.

**Decision (resolved):** auto-apply the course + keyword tagging across all 474 docs, then hand spot-review the two noisy courses (`yannic-kilcher-channel`, `deeplearningai`). Reversible since it only appends a `tags:` field.

| Course directory (real) | Docs | Base topic tags | `level` |
|---|---:|---|---|
| `3blue1brown-neural-networks` | 9 | `topic/neural-network-foundations` (+ `topic/transformers-attention` ch5–6) | intro |
| `andrej-karpathy-channel` | 15 | `topic/neural-network-foundations`, `topic/language-models` | intermediate |
| `andrej-karpathy-neural-networks-zero-to-` | 10 | `topic/neural-network-foundations`, `topic/language-models` | intermediate |
| `deeplearningai` | 49 | `topic/ml-engineering`; `topic/reasoning-agents` by keyword | frontier |
| `jeremy-howard-fastai-practical-deep-lear` | 48 | `topic/generative-models` (SD course); `topic/ml-engineering` for tooling — keyword may override | intermediate |
| `mit-6s191-introduction-to-deep-learning-` | 86 | `topic/neural-network-foundations`; applied/ethics guests → `topic/ai-industry-news` | intro |
| `stanford-cs224n-nlp-with-deep-learning-w` | 46 | `topic/sequence-models-rnn`, `topic/transformers-attention` | intermediate |
| `stanford-cs229-machine-learning-andrew-n` | 20 | `topic/classical-ml` | intermediate |
| `stanford-cs230-deep-learning-andrew-ng-2` | 9 | `topic/neural-network-foundations` | intermediate |
| `stanford-cs231n-cnns-for-visual-recognit` | 14 | `topic/computer-vision` | intermediate |
| `stanford-cs236-deep-generative-models` | 15 | `topic/generative-models` | advanced |
| `stanford-cs25-transformers-united` | 39 | `topic/transformers-attention` | advanced |
| `stanford-cs336-language-modeling-from-sc` | 15 | `topic/language-models`, `topic/efficiency-systems` | advanced |
| `yannic-kilcher-channel` | 99 | **No blanket base** — driven entirely by title keyword; ML-News/drama → `topic/ai-industry-news` | frontier |

**Title-keyword overrides** (case-insensitive, additive, applied after base):
`transformer|attention`→transformers-attention; `diffusion|stable diffusion|score|flow matching|text-to-image`→generative-models; `GAN|VAE|autoencoder`→generative-models; `convolutional|CNN`→computer-vision; `RNN|LSTM|RWKV|xLSTM`→sequence-models-rnn; `mamba|state space|SSM`→efficient-architectures; `MoE|mixtral|switch`→language-models (+technique/moe); `GPT|LLaMA|tokeniz|pretrain`→language-models; `reasoning|chain of thought|tree of thoughts|test-time|GRPO`→reasoning-agents; `RLHF|DPO|ORPO|alignment|SFT|preference`→alignment-rlhf; `Q-learning|MDP|policy gradient|reward`→reinforcement-learning; `agent|agentic|tool|MCP`→reasoning-agents / ml-engineering; `retrieval|RAG`→reasoning-agents (+technique/rag); `quantiz|parallelism|GPU|kernel|triton|inference|serving`→efficiency-systems; `circuits|induction heads|ROME|interpretab`→interpretability; `[ML News]|fired|hacked|NeurIPS … Session|sentient`→ai-industry-news (suppresses other tags).

The two messy courses (`yannic-kilcher-channel`, `deeplearningai`) get ~70% automated coverage and a manual spot-review pass.

---

## 5. Obsidian layer

All new navigation content lives in a new top-level `atlas/` folder plus a bundled `.obsidian/`. `corpus/` is untouched except for the additive frontmatter above.

```
machine-learning-library/
├── README.md                     ← EDIT: append a short "Open in Obsidian / Connect your agent" section
├── .gitignore                    ← EDIT: ignore .obsidian/workspace*.json, cache, plugins/*/data.json
├── AGENTS.md                     ← NEW: cross-tool agent instructions (see §6)
├── CLAUDE.md                     ← NEW: Claude Code instructions (copy of AGENTS.md)
├── .claude/skills/ml-library/SKILL.md   ← NEW: Claude Code skill (see §6)
├── .obsidian/                    ← NEW: bundled, committed vault config
│   ├── app.json  appearance.json  core-plugins.json  community-plugins.json
│   ├── graph.json                ← color groups by topic tag
│   └── plugins/{dataview, obsidian-front-matter-title-plugin}/
├── corpus/                       ← UNCHANGED paths (frontmatter gets +tags/+aliases only)
│   ├── INDEX.md                  ← UNCHANGED (kept for back-compat; superseded by atlas/Home.md)
│   └── papers/ youtube/ web/     ← UNCHANGED
└── atlas/                        ← NEW: the entire navigation layer (~40 files)
    ├── Home.md                   ← root MOC / entry point
    ├── README.md                 ← "what is atlas/, how to use it" (GitHub-facing)
    ├── TAGS.md                   ← canonical tag-schema doc (the §3 vocabulary)
    ├── topics/                   ← 17 topic MOCs (one per primary tag)
    ├── sources/                  ← Papers.md, Courses & Lectures.md, Web Explainers.md
    │   └── courses/              ← 14 lightweight per-course hubs
    └── paths/                    ← curated learning paths
        ├── Zero to Transformer.md
        ├── Diffusion from Scratch.md
        └── LLM Training & Alignment.md
```

**How docs surface by topic without moving files** — each topic MOC has the same three-part body:

1. **Curated "Start here" wikilinks** using `[[ID|Alias]]` form (e.g. `[[1706.03762|Attention Is All You Need]]`, `[[kCc8FmEb1nY|Karpathy — Let's build GPT]]`). These render and click on github.com, work in vanilla Obsidian with zero plugins, **and draw the graph-view edges** (Dataview lists do not create edges).
2. **A Dataview auto-list** that scans the whole vault by `tags:` and lists every matching doc in place — a paper in `corpus/papers/`, a CS25 lecture in `corpus/youtube/`, and a Jalammar article in `corpus/web/` all appear in one table though they live in three folders. Add a tag to a new doc → it appears automatically. Zero files moved.

   ````
   ```dataview
   TABLE WITHOUT ID link(file.link, default(title, file.name)) AS Document,
     category AS Type, default(published,"—") AS Date
   FROM #topic/transformers-attention
   SORT level ASC, published ASC
   ```
   ````
3. **Cross-links** to adjacent hubs (transformers-attention ↔ efficient-architectures ↔ sequence-models-rnn; language-models ↔ reasoning-agents ↔ alignment-rlhf; generative-models ↔ multimodal).

**Readable opaque filenames — two complementary, additive fixes:**
- `aliases:` (vanilla, works everywhere): typing `[[Bahdanau` autocompletes; alias indexed in search. Derived mechanically from existing `title:`.
- **Front Matter Title plugin** (bundled): relabels the file explorer *and graph view* nodes to show `title:` instead of `1409.0473.md`. Reads existing `title:`; requires no file edits.

**Bundled config:** `community-plugins.json` ships exactly two plugins — **Dataview** (auto-lists) and **Front Matter Title** (readable graph). `graph.json` pre-colors topic groups so the global graph opens as a topic-clustered map with `atlas/` MOCs as high-degree hubs. `.gitignore` excludes only machine-specific cache/workspace files.

---

## 6. Connect-your-agent

**Recommended primary path (zero end-user setup): bundle in-repo instructions + a Claude Code skill, and let users open the folder directly.** Best fit for the developer crowd driving the star momentum: `git clone`, open in Cursor / Claude Code / Codex, done.

**Bundled in the repo (all additive, path-safe):**
- **`AGENTS.md`** (repo root) — the cross-tool standard read natively by Cursor, Codex, Copilot, Gemini CLI, Aider, Windsurf, Zed. Describes the `corpus/{papers,youtube,web}` layout, that `corpus/INDEX.md` is the master index, the topic vocabulary, the retrieval workflow (search → read top 3–5 → cite frontmatter `url`), and the frontmatter schema. **Important caveat to state explicitly:** after backfill, YouTube docs are tagged — but agents should still full-text search when tag filters return little.
- **`CLAUDE.md`** — same content for Claude Code (which doesn't read `AGENTS.md`); copy.
- **`.claude/skills/ml-library/SKILL.md`** — a project-scoped Claude Code skill encoding the multi-step retrieval procedure, injected on demand (saves context vs always-on CLAUDE.md).

**Fallback 1 — Obsidian users who want write-back:** the **Local REST API** community plugin (ships a built-in MCP server at `https://127.0.0.1:27124/mcp/`). Lets the agent read the active note, patch frontmatter, and create new notes (synthesized study notes / MOCs) into the vault. Connect via `claude mcp add --transport http obsidian …`.

**Fallback 2 — Claude Desktop / semantic quality:**
- **Filesystem MCP** (`@modelcontextprotocol/server-filesystem` pointed at the cloned folder) for Claude Desktop without Obsidian.
- **RAG quickstart** at `examples/rag_quickstart.py` (sentence-transformers `bge-small-en-v1.5` → embedded LanceDB, chunk-by-heading, returns top-k chunks + `url` for citation). The only option giving true semantic retrieval over the 10M-token corpus; exposable as a one-tool `search_ml_library` MCP server.

**One-line README decision guide:** open in Cursor/Claude Code → nothing to do · live in Obsidian + want read/write → Local REST API plugin · want semantic search → run the RAG quickstart.

---

## 7. What we ADD vs leave untouched

**ADD (new):**
- Frontmatter fields on existing docs: `tags:` and `aliases:` (and only those).
- New folder `atlas/` (~40 hub/MOC/path files) incl. `atlas/README.md`, `atlas/TAGS.md`.
- New folder `.obsidian/` (bundled vault config + 2 vendored plugins).
- New root files `AGENTS.md`, `CLAUDE.md`, `.claude/skills/ml-library/SKILL.md`.
- New `examples/rag_quickstart.py` (optional semantic layer).
- Edits to `README.md` (append a section) and `.gitignore` (append ignores).

**LEAVE UNTOUCHED:**
- Every existing file path under `corpus/` (78 papers, 474 youtube, 38 web) — no moves, no renames.
- The existing `topics:` field and all other existing frontmatter fields.
- `corpus/INDEX.md` (kept for back-compat / existing RAG pointers).
- All file *bodies* (frontmatter additions only; no body edits) — except the LF normalization in §8 step 1, which is semantically null.
- The 9 duplicate Karpathy files stay duplicated in place (tagged identically).

---

## 8. Build order (phased, each step independently shippable & reversible)

1. **Encoding/format hygiene (prereq, approved).** Normalize line endings to LF across the corpus (the audit found ~all papers use CRLF, which breaks line-anchored grep/YAML tooling and masked the true topic count). Semantically null. Also normalize `topics:` array spacing to canonical `["a", "b"]`. Do this first or the tagging script misfires.
2. **Aliases pass.** Mechanically derive `aliases:` from existing `title:` on all 590 docs → instant readable links, zero risk.
3. **MOC scaffold.** Write `atlas/Home.md`, `atlas/TAGS.md`, and the 17 topic MOCs with curated wikilinks only. Fully functional vanilla-Obsidian + GitHub navigation with no plugins and no tagging yet.
4. **Tag injection.** Idempotent script that only ever appends `tags:` (never edits `topics:`): papers/web mapped from existing `topics:`; the 474 YouTube docs via course-prior base + title-keyword override (§4 tables); 9 Karpathy duplicates tagged identically. Spot-review yannic + deeplearningai by hand. Dataview blocks now light up.
5. **Bundle `.obsidian/`.** Dataview + Front Matter Title + graph colors + `.gitignore` rules.
6. **Agent connection.** Add `AGENTS.md`, `CLAUDE.md`, `.claude/skills/ml-library/SKILL.md`; append the README guide.
7. **Optional power-user.** `atlas/paths/*` learning paths, per-course hubs, `examples/rag_quickstart.py`.

Each step is independently reversible: delete `atlas/`/`.obsidian/` and revert the YAML adds → today's repo.

---

## 9. Resolved decisions

1. **`frontier` tag** → reclassified to the `level/frontier` facet; dropped from topic vocabulary. Topics stay subject-based.
2. **YouTube backfill autonomy** → auto-apply course + keyword tags to all 474 docs, then spot-review `yannic-kilcher-channel` + `deeplearningai`.
3. **Catch-all topics** → `ml-engineering` (~60) and `ai-industry-news` (~45) accepted as-is.
4. **Encoding fix** → LF normalization approved as build step 1.

---

## 10. Out of scope

- No physical reorganization of `corpus/` into topic folders (explicitly rejected — would break paths and force one-topic-per-file).
- No edits to document *bodies* beyond LF normalization.
- No removal/rewrite of the existing `topics:` field.
- No new original educational content (the repo remains a curation/reformatting of others' work).
- Building/hosting a live hosted RAG service (the `rag_quickstart.py` is a local example only).

---

## 11. Success criteria

- All 590 docs carry controlled `tags:` and `aliases:`; `topics:` and all other existing fields and paths are byte-identical except LF normalization.
- Opening the repo as an Obsidian vault shows a topic-clustered graph with readable node titles and 17 working topic hubs, each auto-listing its docs via Dataview.
- On github.com, every topic MOC renders as clickable titled lists (no broken links).
- A user can `git clone`, open in Claude Code / Cursor, and have the agent retrieve and cite corpus docs using only the bundled `AGENTS.md`/`CLAUDE.md`/skill.
- Deleting `atlas/` + `.obsidian/` + the added agent files and reverting the YAML additions restores today's repo exactly.
