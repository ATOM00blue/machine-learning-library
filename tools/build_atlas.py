#!/usr/bin/env python3
"""Generate the atlas/ navigation layer from the tagged corpus.

Reads each doc's frontmatter (title, tags, url, published) and writes:
  atlas/Home.md            - root Map of Content
  atlas/TAGS.md            - the controlled tag vocabulary (with live counts)
  atlas/README.md          - what atlas/ is, for GitHub readers
  atlas/topics/<slug>.md   - one hub per topic: curated start-here + Dataview auto-list + cross-links
  atlas/sources/*.md       - by-source hubs (papers / lectures / web)
  atlas/paths/*.md         - curated learning paths

Links are standard relative Markdown (clickable on GitHub AND in Obsidian, and
they draw graph edges). Dataview blocks add live auto-lists inside Obsidian.
Nothing in corpus/ is moved or renamed.
"""
from __future__ import annotations
import glob, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
ATLAS = os.path.join(ROOT, "atlas")

LEVEL_RANK = {"intro": 0, "intermediate": 1, "advanced": 2, "frontier": 3}
MEDIUM_LABEL = {"paper": "📄 paper", "lecture": "🎓 lecture", "article": "📝 article"}

TOPICS = [
    ("neural-network-foundations", "Neural Network Foundations",
     "MLPs, the forward pass, backpropagation, and the training machinery — optimizers (SGD/Adam), normalization, dropout, initialization. The entry point.",
     ["classical-ml", "computer-vision", "sequence-models-rnn"]),
    ("classical-ml", "Classical & Statistical ML",
     "Non-neural methods: regression, SVMs/kernels, trees & ensembles, naive Bayes, EM, PCA, and learning theory.",
     ["neural-network-foundations"]),
    ("computer-vision", "Computer Vision",
     "CNNs (ResNet, VGG, DenseNet), image classification, object detection & segmentation, and vision transformers (ViT, DETR, MAE).",
     ["neural-network-foundations", "generative-models", "multimodal"]),
    ("sequence-models-rnn", "Sequence Models & Classic NLP",
     "Word embeddings, RNN/LSTM/GRU, seq2seq with attention, neural machine translation, and classic NLP tasks. The bridge to transformers.",
     ["neural-network-foundations", "transformers-attention", "language-models"]),
    ("transformers-attention", "Transformers & Attention",
     "The transformer architecture and attention itself: self-/multi-head attention, positional encodings, and architecture explainers.",
     ["sequence-models-rnn", "language-models", "efficient-architectures", "interpretability"]),
    ("language-models", "Language Models & Pretraining",
     "Building LLMs end-to-end: pretraining, the BERT/GPT/T5/LLaMA model reports, scaling laws, emergent abilities, and mixture-of-experts.",
     ["transformers-attention", "reasoning-agents", "alignment-rlhf", "efficiency-systems"]),
    ("efficient-architectures", "Efficient & Long-Context Architectures",
     "Alternatives to dense quadratic attention: sparse/linear attention, FlashAttention, long-context methods, and state-space models (Mamba, RWKV).",
     ["transformers-attention", "efficiency-systems"]),
    ("generative-models", "Generative Models",
     "Deep generative modeling: VAEs, GANs, normalizing flows, autoregressive models, and diffusion incl. latent/stable diffusion and text-to-image.",
     ["computer-vision", "multimodal"]),
    ("multimodal", "Multimodal & Vision-Language",
     "Models bridging modalities: contrastive vision-language (CLIP), text-to-image, audio/speech transformers, and general multimodal work.",
     ["computer-vision", "generative-models", "language-models"]),
    ("reinforcement-learning", "Reinforcement Learning",
     "RL as a subfield: MDPs, value/policy iteration, Q-learning, policy gradients, deep RL, imitation, and game-playing agents.",
     ["alignment-rlhf"]),
    ("alignment-rlhf", "Alignment, RLHF & Preference Tuning",
     "Post-training LLMs to human preferences: instruction tuning/SFT, RLHF/InstructGPT, reward modeling, DPO/ORPO/GRPO, and safety alignment.",
     ["language-models", "reinforcement-learning", "reasoning-agents"]),
    ("reasoning-agents", "Reasoning & Agents",
     "Eliciting and structuring reasoning, and tool-using agents: chain-of-thought, tree-of-thoughts, test-time compute, ReAct/Toolformer, and RAG.",
     ["language-models", "alignment-rlhf", "ml-engineering"]),
    ("efficiency-systems", "Efficiency, Systems & Serving",
     "Making models cheap to adapt, train, and serve: quantization, LoRA/PEFT, CUDA/Triton kernels, parallelism, and paged-attention serving (vLLM).",
     ["language-models", "efficient-architectures"]),
    ("interpretability", "Interpretability & Model Analysis",
     "Reverse-engineering trained models: mechanistic interpretability (circuits, superposition, induction heads), probing, and knowledge editing.",
     ["transformers-attention", "language-models"]),
    ("evaluation-trust", "Evaluation, Benchmarks & Trust",
     "Measuring and stress-testing models: benchmarks, evaluation methodology, calibration/uncertainty, robustness, bias/fairness, and hallucination.",
     ["language-models", "alignment-rlhf"]),
    ("ml-engineering", "ML Engineering & Tooling",
     "Hands-on build & ship: production agent engineering, app/agent frameworks, dev tooling (FastHTML, nbdev, CUDA-for-Python), and prompt engineering.",
     ["reasoning-agents", "ai-industry-news"]),
    ("ai-industry-news", "AI Industry, News & Meta",
     "News, commentary, and the broader ecosystem: model-release roundups, policy, AGI debate, conference coverage, and applied 'AI for X' talks.",
     ["ml-engineering"]),
]
TOPIC_LABEL = {slug: label for slug, label, _d, _n in TOPICS}

PATHS = [
    ("zero-to-transformer", "Zero to Transformer",
     "From 'what is a neural network' to building a GPT — the fastest honest path through the fundamentals to modern transformers.",
     ["But what is a neural network", "Gradient descent, how neural networks learn",
      "Backpropagation, intuitively", "Efficient Estimation of Word Representations",
      "Sequence to Sequence Learning", "Neural Machine Translation by Jointly Learning",
      "Attention Is All You Need", "Illustrated Transformer", "Let's build GPT"]),
    ("diffusion-from-scratch", "Diffusion from Scratch",
     "The generative-modeling lineage that leads to Stable Diffusion, oldest ideas first.",
     ["Auto-Encoding Variational Bayes", "Generative Adversarial",
      "What are Diffusion Models", "Denoising Diffusion Probabilistic",
      "Denoising Diffusion Implicit", "High-Resolution Image Synthesis with Latent Diffusion",
      "Illustrated Stable Diffusion"]),
    ("llm-training-and-alignment", "LLM Training & Alignment",
     "How a modern chat model is actually built: pretraining at scale, then aligned to human preferences.",
     ["Let's reproduce GPT-2", "Language Models are Few-Shot Learners",
      "Scaling Laws for Neural Language Models", "Training Compute-Optimal Large Language Models",
      "LoRA", "Training language models to follow instructions",
      "Direct Preference Optimization"]),
]


def parse_doc(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    lines = text.split("\n")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), 1)
    fm = {}
    for ln in lines[1:end]:
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", ln)
        if m:
            fm[m.group(1)] = m.group(2)
    title = (fm.get("title", os.path.basename(path))).strip().strip('"')
    tags = []
    if fm.get("tags"):
        tags = [t.strip() for t in fm["tags"].strip("[]").split(",") if t.strip()]
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    year = None
    if fm.get("published"):
        mm = re.search(r"(\d{4})", fm["published"])
        if mm:
            year = int(mm.group(1))
    topics = [t.split("/", 1)[1] for t in tags if t.startswith("topic/")]
    level = next((t.split("/", 1)[1] for t in tags if t.startswith("level/")), "intermediate")
    medium = next((t.split("/", 1)[1] for t in tags if t.startswith("medium/")), "lecture")
    return dict(title=title, rel=rel, url=fm.get("url", "").strip().strip('"'),
                topics=topics, level=level, medium=medium, year=year,
                base=os.path.splitext(os.path.basename(path))[0])


def link(doc, depth):
    """Relative markdown link from a file `depth` dirs below repo root."""
    return f"[{doc['title'].replace('[', '(').replace(']', ')')}]({'../' * depth}{doc['rel']})"


def start_here(docs):
    """Pick a gentle on-ramp: a couple intro lectures/articles + foundational papers."""
    gentle = sorted([d for d in docs if d["medium"] in ("lecture", "article")
                     and d["level"] in ("intro", "intermediate")],
                    key=lambda d: (LEVEL_RANK[d["level"]], d["title"].lower()))[:2]
    papers = sorted([d for d in docs if d["medium"] == "paper"],
                    key=lambda d: (d["year"] or 9999))[:4]
    picked, seen = [], set()
    for d in gentle + papers:
        if d["rel"] not in seen:
            picked.append(d); seen.add(d["rel"])
    if not picked:  # topics with no papers/intro (news/eng) — take a few by name
        picked = sorted(docs, key=lambda d: d["title"].lower())[:4]
    return picked[:6]


def main():
    docs = [parse_doc(p) for p in glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True)
            if os.path.basename(p) != "INDEX.md"]
    by_topic = defaultdict(list)
    for d in docs:
        for t in d["topics"]:
            by_topic[t].append(d)

    os.makedirs(os.path.join(ATLAS, "topics"), exist_ok=True)
    os.makedirs(os.path.join(ATLAS, "sources"), exist_ok=True)
    os.makedirs(os.path.join(ATLAS, "paths"), exist_ok=True)

    # ---- topic MOCs --------------------------------------------------------
    for slug, label, desc, neighbors in TOPICS:
        tdocs = by_topic.get(slug, [])
        out = [f"---", f'title: "{label}"', f"aliases: [\"{label}\"]",
               f"cssclasses: [moc]", "---", "", f"# {label}", "", f"> {desc}", "",
               f"*{len(tdocs)} documents.* ", "",
               "## Start here", ""]
        for i, d in enumerate(start_here(tdocs), 1):
            out.append(f"{i}. {link(d, 2)}  · {MEDIUM_LABEL.get(d['medium'], d['medium'])} · {d['level']}")
        out += ["", "## All documents", "",
                "```dataview", "TABLE WITHOUT ID",
                "  link(file.link, default(title, file.name)) AS Document,",
                '  default(source, "") AS Type,',
                '  default(published, "") AS Date',
                f'FROM #topic/{slug} and -"atlas"',
                "SORT level ASC, published ASC", "```", "",
                "_(The list above renders in Obsidian with the Dataview plugin. On GitHub, browse **Start here** or the [full index](../../corpus/INDEX.md).)_",
                "", "## Related topics", ""]
        out.append(" · ".join(f"[{TOPIC_LABEL[n]}]({n}.md)" for n in neighbors))
        out += ["", "---", "", "[← Atlas home](../Home.md)"]
        open(os.path.join(ATLAS, "topics", f"{slug}.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")

    # ---- Home --------------------------------------------------------------
    h = ["---", 'title: "ML Library — Home"', 'aliases: ["Home", "Atlas", "Start here"]',
         "cssclasses: [moc]", "---", "",
         "# 🧠 Machine Learning Library — Atlas", "",
         f"A topic map over **{len(docs)} documents** — papers, lectures, and articles — "
         "all tagged and cross-linked. Open any hub to see every matching doc, wherever it lives.",
         "", "## Topics", "",
         "| Topic | Docs | |", "|---|--:|---|"]
    for slug, label, desc, _n in TOPICS:
        h.append(f"| [{label}](topics/{slug}.md) | {len(by_topic.get(slug, []))} | {desc.split('.')[0]}. |")
    h += ["", "## Learning paths", ""]
    for pslug, plabel, pdesc, _steps in PATHS:
        h.append(f"- [{plabel}](paths/{pslug}.md) — {pdesc}")
    h += ["", "## By source", "",
          "- [Papers](sources/papers.md) · [Lectures](sources/lectures.md) · [Web explainers](sources/web.md)",
          "- [Full machine-generated index](../corpus/INDEX.md)", "",
          "---", "",
          "*New here? Start with [Zero to Transformer](paths/zero-to-transformer.md). "
          "Connecting an agent? See [`AGENTS.md`](../AGENTS.md).*"]
    open(os.path.join(ATLAS, "Home.md"), "w", encoding="utf-8").write("\n".join(h) + "\n")

    # ---- TAGS.md -----------------------------------------------------------
    t = ["# Tag vocabulary", "",
         "Every doc carries a controlled `tags:` list (layered on top of the original "
         "free-form `topics:` field, which is left untouched). Five namespaces:", "",
         "## `topic/` — subject (17, multi-valued)", "", "| Tag | Docs | Covers |", "|---|--:|---|"]
    for slug, label, desc, _n in TOPICS:
        t.append(f"| `topic/{slug}` | {len(by_topic.get(slug, []))} | {desc} |")
    t += ["", "## `level/` — difficulty", "",
          "`level/intro` · `level/intermediate` · `level/advanced` · `level/frontier`", "",
          "## `medium/` — format", "",
          "`medium/paper` · `medium/lecture` · `medium/article`", "",
          "## `task/` — modality / problem domain", "",
          "`task/vision` · `task/language` · `task/speech-audio` · `task/multimodal` · "
          "`task/graph` · `task/rl-control` · `task/tabular-classical` · `task/general`", "",
          "## `technique/` — architecture / method", "",
          "`technique/mlp` · `technique/cnn` · `technique/rnn-lstm` · `technique/transformer` · "
          "`technique/attention` · `technique/diffusion` · `technique/gan` · `technique/vae` · "
          "`technique/normalizing-flow` · `technique/ssm` · `technique/moe` · `technique/lora-peft` · "
          "`technique/quantization` · `technique/rlhf` · `technique/dpo` · `technique/ppo` · "
          "`technique/cot` · `technique/rag` · `technique/flashattention` · `technique/embeddings`", "",
          "---", "",
          "Tags are auto-assigned (keyword rules + a content-reading pass); they're great for "
          "narrowing, but full-text search is the ground truth when a filter looks sparse."]
    open(os.path.join(ATLAS, "TAGS.md"), "w", encoding="utf-8").write("\n".join(t) + "\n")

    # ---- atlas/README ------------------------------------------------------
    r = ["# atlas/", "",
         "The navigation layer for this corpus. It adds **no new source material** — it links "
         "to the documents in `corpus/`, organized by topic instead of by source.", "",
         "- **[Home.md](Home.md)** — the topic map (start here).",
         "- **[topics/](topics/)** — one hub per subject; each auto-lists every matching doc.",
         "- **[paths/](paths/)** — curated reading orders.",
         "- **[TAGS.md](TAGS.md)** — the controlled tag vocabulary.", "",
         "Best experienced by opening the repo as an [Obsidian](https://obsidian.md) vault "
         "(a bundled config + plugins make the hubs auto-populate and the graph cluster by topic), "
         "but every page is plain Markdown and works on GitHub too."]
    open(os.path.join(ATLAS, "README.md"), "w", encoding="utf-8").write("\n".join(r) + "\n")

    # ---- source hubs -------------------------------------------------------
    for key, fname, label in [("paper", "papers.md", "Papers"),
                              ("lecture", "lectures.md", "Lectures"),
                              ("article", "web.md", "Web explainers")]:
        sd = sorted([d for d in docs if d["medium"] == key], key=lambda d: d["title"].lower())
        s = [f"# {label}  ({len(sd)})", "",
             f"```dataview", "TABLE WITHOUT ID link(file.link, default(title, file.name)) AS Document, "
             "default(published, \"\") AS Date",
             f'FROM #medium/{key} and -"atlas"', "SORT title ASC", "```", "",
             "_Rendered list needs Obsidian + Dataview; the static list below works everywhere._", ""]
        for d in sd:
            s.append(f"- {link(d, 2)}")
        s += ["", "[← Atlas home](../Home.md)"]
        open(os.path.join(ATLAS, "sources", fname), "w", encoding="utf-8").write("\n".join(s) + "\n")

    # ---- learning paths ----------------------------------------------------
    missing = []
    for pslug, plabel, pdesc, steps in PATHS:
        p = ["---", f'title: "{plabel}"', f'aliases: ["{plabel}"]', "cssclasses: [moc]", "---", "",
             f"# {plabel}", "", f"> {pdesc}", ""]
        n = 0
        for q in steps:
            hit = next((d for d in docs if q.lower() in d["title"].lower()), None)
            if hit:
                n += 1
                p.append(f"{n}. {link(hit, 2)}  · {MEDIUM_LABEL.get(hit['medium'], hit['medium'])}")
            else:
                missing.append((pslug, q))
        p += ["", "---", "", "[← Atlas home](../Home.md)"]
        open(os.path.join(ATLAS, "paths", f"{pslug}.md"), "w", encoding="utf-8").write("\n".join(p) + "\n")

    print(f"atlas built: Home + TAGS + README + {len(TOPICS)} topics + 3 sources + {len(PATHS)} paths")
    if missing:
        print("  unresolved path steps (title not found):")
        for pslug, q in missing:
            print(f"    [{pslug}] {q}")


if __name__ == "__main__":
    main()
