# Custom Machine Learning Algorithm + Curated ML Knowledge Base

**One repository with two connected layers: Adaptive Prototype Residual Learning
(APRL), a custom experimental algorithm for numeric tabular ML, and a
hand-curated, machine-readable knowledge base of leading university courses,
canonical research papers, and widely cited explainers—normalized into consistent
Markdown with full provenance.**

Custom classifier + regressor · 923 documents · ~11 million tokens · beginner to
frontier (2026) research · every source credited.

![docs](https://img.shields.io/badge/documents-923-blue)
![tokens](https://img.shields.io/badge/tokens-~11M-green)
![papers](https://img.shields.io/badge/arXiv_papers-391-orange)
![lectures](https://img.shields.io/badge/lectures-474-red)
![articles](https://img.shields.io/badge/articles-58-purple)
![topics](https://img.shields.io/badge/topics-17-blueviolet)
![tests](https://img.shields.io/badge/APRL_tests-11_passing-brightgreen)

> **🧪 Build with APRL.** Train the repository's custom classifier or regressor,
> inspect uncertainty, and reproduce its baseline comparisons. →
> [**Explore the algorithm**](algorithms/aprl/README.md)
>
> **📚 Learn and retrieve from the knowledge base.** Every document is tagged into
> a 17-topic map; open the repository as an Obsidian vault or point an AI agent at
> it for answers grounded in papers and lectures. →
> [**Open in Obsidian / connect your agent**](#open-in-obsidian--connect-your-agent)

---

## Why this exists

Machine learning repositories often provide either code without enough conceptual
context or learning material without a concrete system to experiment with. This
project keeps both in one place: APRL provides an inspectable learning algorithm,
while the curated corpus provides the theory, provenance, and educational context
around it.

The best material for learning machine learning is scattered across course pages,
YouTube channels, arXiv PDFs, and personal blogs. The knowledge-base layer pulls
high-signal sources into **one place**, in **one consistent format**, with **clean
metadata on every file**. Instead of an undifferentiated dump, it is a deliberately
chosen collection spanning fundamentals through frontier research.

Use the repository to train and evaluate APRL, study how the algorithm is built,
read the corpus directly, build retrieval systems, benchmark embeddings, or ground
an AI tutor in attributed sources.

---

## At a glance

| | |
|---|---|
| **Custom algorithm** | APRL classifier + regressor for numeric tabular data |
| **Algorithm capabilities** | Local nonlinear residuals, uncertainty, missing values, incremental batches |
| **Algorithm evidence** | 11 tests + reproducible five-dataset smoke benchmark |
| **Total documents** | 923 |
| **Total size** | ~42M characters (~11M tokens) |
| **arXiv papers** | 391 (78 full-text + 313 recent abstract+metadata) |
| **Lecture transcripts** | 474 (across 14 courses/channels) |
| **Web articles** | 58 (canonical explainers) |
| **Topic tags** | 17-topic controlled vocabulary (+ level / medium / task / technique facets) |
| **Format** | Markdown + YAML frontmatter |
| **Coverage** | Intro fundamentals → frontier 2025–2026 research |

Every file begins with structured frontmatter (title, source, URL, authors, date, topics, **controlled `tags`**, `aliases`) so the whole corpus is trivially filterable and parseable — and navigable by topic (see [`atlas/`](atlas/) and [`atlas/TAGS.md`](atlas/TAGS.md)).

---

## What's inside

### Custom ML algorithm (`src/aprl/`)

**Adaptive Prototype Residual Learning (APRL)** combines a global ridge model
with relevance-weighted prototypes and local polynomial experts that correct the
global model's residual errors. It provides classification, regression,
distance-aware uncertainty, missing-value handling, and exact incremental batch
updates through a NumPy-only runtime. The implementation is backed by tests,
benchmarks, mathematical documentation, and a conservative related-work review.

APRL is a custom experimental composition, not a claim of scientific novelty or
state-of-the-art performance. See the [algorithm guide](algorithms/aprl/README.md)
and [reproducible results](algorithms/aprl/BENCHMARKS.md).

### Research papers (`corpus/papers/` — 391)

The 78 canonical papers in **full text**, plus **313 recent (2024H2–2026)** papers added as the **verbatim abstract + metadata** (with a link to the full paper). Foundational to frontier:

- **Foundations** — Dropout, word2vec, Seq2Seq, Adam, Batch/Layer Norm, VAE, GANs
- **Vision** — VGG, GoogLeNet, ResNet, DenseNet, Faster R-CNN, YOLOv3, ViT, MAE, DETR
- **The Transformer era** — *Attention Is All You Need*, BERT, RoBERTa, T5, GPT-3, Chinchilla, scaling laws
- **Efficient attention & serving** — FlashAttention 1/2/3, Linformer, Longformer, Performer, Reformer, PagedAttention/vLLM, MQA/GQA, RoPE/ALiBi
- **Generative models** — DDPM, DDIM, Latent Diffusion (Stable Diffusion), DALL·E 2, DiT, VQ-VAE, StyleGAN, CLIP
- **LLMs & alignment** — LLaMA 1/2, Mistral, Mixtral, InstructGPT/RLHF, DPO, LoRA/QLoRA/DoRA, GPTQ/AWQ/LLM.int8()
- **Reasoning & agents** — Chain-of-Thought, Self-Consistency, Tree of Thoughts, ReAct, Toolformer
- **Frontier (2024–2025)** — Mamba, State-Space Duality, Mixture-of-Depths, DeepSeek V2/V3/R1, Native Sparse Attention
- **Newly added (2025–2026)** — Titans, RWKV-7, Gated DeltaNet, Mamba-3 & hybrid linear attention; reasoning & test-time compute (RLVR, GRPO-line); BitNet / FP4 quantization; diffusion language models; video/image generation; VLMs; LLM agents & RAG; SAE / attribution-graph interpretability; world models; vision-language-action robotics; frontier model reports; AI-for-science

### Lecture transcripts (`corpus/youtube/` — 474)

Full transcripts from the most respected ML courses and educators:

| Course / Channel | Lectures |
|---|---:|
| MIT 6.S191 — Introduction to Deep Learning | 86 |
| Yannic Kilcher — paper walkthroughs | 99 |
| DeepLearning.AI | 49 |
| fast.ai — Practical Deep Learning (Jeremy Howard) | 48 |
| Stanford CS224n — NLP with Deep Learning | 46 |
| Stanford CS25 — Transformers United | 39 |
| Stanford CS229 — Machine Learning (Andrew Ng) | 20 |
| Stanford CS336 — Language Modeling from Scratch | 15 |
| Stanford CS236 — Deep Generative Models | 15 |
| Stanford CS231n — CNNs for Visual Recognition | 14 |
| Stanford CS230 — Deep Learning (Andrew Ng) | 9 |
| Andrej Karpathy — channel + Neural Networks: Zero to Hero | 25 |
| 3Blue1Brown — Neural Networks series | 9 |

### Web articles (`corpus/web/` — 58)

The explainers practitioners actually link to: Jay Alammar's *Illustrated* series, Lilian Weng's deep-dives, Sebastian Raschka, the Stanford CS231n notes, *Dive into Deep Learning*, Distill.pub, Anthropic's Transformer Circuits, and Karpathy's blog.

---

## Repository structure

```
machine-learning-library/
├── README.md                  ← you are here
├── SOURCES.md                 ← full attribution: every source, credited
├── NOTICE.md                  ← licensing & usage notes
├── AGENTS.md / CLAUDE.md      ← how an AI agent should navigate & cite this corpus
├── corpus/
│   ├── INDEX.md               ← machine-generated index of all 923 files
│   ├── papers/                ← 391 arXiv papers (78 full-text + 313 abstract+metadata)
│   ├── youtube/               ← 474 lecture transcripts, grouped by channel
│   └── web/                   ← 58 articles, grouped by domain
├── atlas/                     ← topic navigation layer (Maps of Content + learning paths)
│   ├── Home.md                ← start here when browsing in Obsidian
│   ├── TAGS.md                ← the controlled tag vocabulary
│   ├── topics/                ← one hub per topic (auto-lists every matching doc)
│   └── paths/                 ← curated reading paths (Zero to Transformer, …)
├── .obsidian/                 ← bundled vault config — open the folder in Obsidian and it just works
├── tools/                     ← scripts that clean, tag, and index the corpus
├── algorithms/aprl/           ← APRL specification, benchmarks, and related work
├── src/aprl/                  ← installable APRL classifier and regressor
├── tests/                     ← APRL behavioral test suite
├── benchmarks/                ← reproducible model comparisons
└── examples/
    ├── self-attention-study-note.md   ← a synthesized, fully-cited study note
    ├── rag_quickstart.py              ← minimal semantic search / RAG over the corpus
    └── aprl_quickstart.py             ← APRL classification/regression example
```

### Frontmatter format

Every document looks like this:

```markdown
---
title: "Attention Is All You Need"
source: "arxiv"
arxiv_id: "1706.03762"
url: "http://arxiv.org/abs/1706.03762v7"
authors: ["Ashish Vaswani", "Noam Shazeer", ...]
published: "2017-06-12"
topics: ["transformer", "attention"]          # original free-form tags
aliases: ["Attention Is All You Need"]         # readable Obsidian wikilink targets
tags: [topic/transformers-attention, level/advanced, medium/paper, task/language, technique/attention]
---

## Abstract
...
## Full Text
...
```

`tags:` is a controlled, queryable vocabulary (topic / level / medium / task /
technique) layered on top of the original `topics:` — see [`atlas/TAGS.md`](atlas/TAGS.md).

---

## Use cases

This corpus is a building block. Some of the things it's good for:

1. **Retrieval-augmented ML tutor.** Embed the corpus, drop it in a vector DB, and build a Q&A assistant that answers ML questions grounded in real sources — and can cite the exact lecture or paper it drew from. No hallucinated references.

2. **Fine-tuning a domain model.** ~11M tokens of clean, on-topic ML text is a realistic dataset for continued-pretraining or instruction-tuning a small (1–7B) "ML explainer" model.

3. **Embeddings / retrieval benchmark.** A coherent, single-domain corpus is ideal for evaluating embedding models and retrieval pipelines on technical content.

4. **Synthesized study notes.** Use an LLM to compress multiple sources on one topic into a single cited note. `examples/self-attention-study-note.md` shows the output: a self-attention explainer assembled from the original paper, two blog posts, and a Karpathy lecture — every claim traced back to its source.

5. **Concept / citation graphs.** The frontmatter + cross-references make it straightforward to extract a graph of which papers and lectures explain which concepts.

6. **Personalized reading paths.** Filter by topic and source to generate an ordered learning path (e.g. "everything on diffusion models, easiest first").

7. **Offline reference library.** It's just Markdown — grep it, open it in Obsidian, read it on a plane.

### Custom machine-learning algorithm: APRL

The repository now also contains **Adaptive Prototype Residual Learning (APRL)**,
a custom experimental algorithm for numeric tabular classification and regression.
It combines a global ridge model, relevance-weighted prototypes, local polynomial
residual experts, distance-aware blending, and uncertainty estimates.

```bash
python -m pip install -e .
python examples/aprl_quickstart.py
```

Start with [`algorithms/aprl/README.md`](algorithms/aprl/README.md) for the full
mathematical specification, limitations, API, tests, related work, and reproducible
baseline benchmarks. APRL is presented as an experimental custom composition—not
as a proven state-of-the-art or scientifically novel method.

---

## Quick start

```bash
git clone https://github.com/ATOM00blue/machine-learning-library.git
cd machine-learning-library

# Browse the full index
less corpus/INDEX.md

# Everything is plain Markdown — search it however you like
grep -rl "flash attention" corpus/

# Filter by frontmatter, e.g. all 2024+ papers (with yq)
for f in corpus/papers/*.md; do
  yq -f extract '.published' "$f" | grep -q '^202[45]' && echo "$f"
done
```

Minimal RAG sketch (Python):

```python
import glob, frontmatter
from sentence_transformers import SentenceTransformer

docs = [frontmatter.load(p) for p in glob.glob("corpus/**/*.md", recursive=True)]
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode([d.content for d in docs])
# ... store in your vector DB of choice and query
```

A ready-to-run version of this lives in [`examples/rag_quickstart.py`](examples/rag_quickstart.py)
(`python examples/rag_quickstart.py --build`, then ask it questions).

---

## Open in Obsidian / connect your agent

This repo is also a ready-to-use **Obsidian vault** and an **agent-friendly
knowledge base** — pick whichever fits how you work:

**📓 Browse it in [Obsidian](https://obsidian.md).** Open the cloned folder as a
vault — a bundled `.obsidian/` config sets up a topic-colored graph and sensible
defaults out of the box.
- **`atlas/Home.md`** is your start page; **`atlas/topics/`** has a hub per topic
  with a curated reading list and cross-links — all working with no plugins.
- Two **optional** community plugins make it shine; Obsidian will offer to enable
  them, or install from *Settings → Community plugins*:
  [Dataview](https://github.com/blacksmithgu/obsidian-dataview) (live auto-listed
  doc tables in each hub) and
  [Front Matter Title](https://github.com/snezhig/obsidian-front-matter-title)
  (graph/explorer nodes show titles instead of arXiv/video IDs).
- Everything degrades gracefully — the hubs, links, tags, and graph also render
  fine on GitHub and as plain Markdown with no plugins at all.

**🤖 Point your AI agent at it.** Choose one:
| You use… | Do this |
|---|---|
| Cursor / Codex / Copilot / Gemini CLI / Aider / Zed | Open the folder — they read [`AGENTS.md`](AGENTS.md) automatically. |
| Claude Code | Open the folder — it reads [`CLAUDE.md`](CLAUDE.md); a `/ml-library` skill is bundled. |
| Claude Desktop | Add a [Filesystem MCP server](https://modelcontextprotocol.io) pointed at this folder. |
| Obsidian + live read/write | Install the **Local REST API** plugin (built-in MCP server) and `claude mcp add` it. |
| Semantic search / RAG | Run [`examples/rag_quickstart.py`](examples/rag_quickstart.py). |

Your agent then answers ML questions grounded in these sources and **cites the
exact paper or lecture** — no hallucinated references.

---

## Attribution

**All credit belongs to the original creators.** This repository is a *curation and reformatting* of publicly available educational material — it contains no original research or teaching content of its own. Every document retains its source URL and authorship in the frontmatter, and **[SOURCES.md](SOURCES.md)** lists every course, channel, publication, and paper included.

If you find this useful, please support the people who actually made the material: subscribe to the channels, take the courses, read the papers, and cite the authors.

See **[NOTICE.md](NOTICE.md)** for licensing details and the takedown/removal policy.

---

## License

- The **structure, index, scripts, APRL implementation, and organization** of this repository are released under the MIT License.
- The **content of each document** remains under the rights of its original author/publisher and is included here for research and educational purposes. See [NOTICE.md](NOTICE.md).
