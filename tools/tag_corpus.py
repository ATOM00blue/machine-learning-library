#!/usr/bin/env python3
"""Additively tag the corpus: append `tags:` and `aliases:` to each doc's
YAML frontmatter WITHOUT touching any existing field (incl. `topics:`).

- medium  : from folder (papers->paper, youtube->lecture, web->article)
- topic   : controlled vocab (17). Papers/web: mapped from existing `topics:`
            plus title keywords. YouTube: course-prior base + title keywords.
- level   : course default / paper date / article default
- task    : derived from topics
- technique: from title keywords + existing topics
- aliases : from `title:` (full title + a short variant when unambiguous)

Idempotent: existing tags:/aliases: lines are replaced, not duplicated.
Emits a JSON manifest (topic -> docs) to tools/.tags_manifest.json for the
atlas MOC builder.
"""
from __future__ import annotations
import glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")
# Optional human/agent overrides for docs that title-keyword tagging gets wrong.
# rel-path -> {"topics": [...], "techniques": [...], "level": "..."}
_OV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overrides.json")
OVERRIDES = json.load(open(_OV_PATH)) if os.path.exists(_OV_PATH) else {}

TOPICS = [
    "neural-network-foundations", "classical-ml", "computer-vision",
    "sequence-models-rnn", "transformers-attention", "language-models",
    "efficient-architectures", "generative-models", "multimodal",
    "reinforcement-learning", "alignment-rlhf", "reasoning-agents",
    "efficiency-systems", "interpretability", "evaluation-trust",
    "ml-engineering", "ai-industry-news",
]

# --- existing free-form topic value -> controlled topic(s) -------------------
TOPIC_MAP = {
    "regularization": "neural-network-foundations", "dropout": "neural-network-foundations",
    "nn": "neural-network-foundations", "mlp": "neural-network-foundations",
    "backprop": "neural-network-foundations", "foundations": "neural-network-foundations",
    "optimization": "neural-network-foundations", "normalization": "neural-network-foundations",
    "batchnorm": "neural-network-foundations", "layernorm": "neural-network-foundations",
    "adam": "neural-network-foundations", "training-practical": "neural-network-foundations",
    "regression": "classical-ml", "classification": "computer-vision",
    "cnn": "computer-vision", "detection": "computer-vision", "vision": "computer-vision",
    "residual": "computer-vision", "transfer-learning": "computer-vision",
    "embeddings": "sequence-models-rnn", "seq2seq": "sequence-models-rnn", "rnn": "sequence-models-rnn",
    "attention": "transformers-attention", "transformer": "transformers-attention",
    "positional": "transformers-attention",
    "scaling": "language-models", "pretraining": "language-models", "bert": "language-models",
    "gpt": "language-models", "moe": "language-models", "self-supervised": "language-models",
    "efficient": "efficient-architectures", "flashattention": "efficient-architectures",
    "sparse": "efficient-architectures", "ssm": "efficient-architectures",
    "long-context": "efficient-architectures", "mla": "efficient-architectures",
    "generative": "generative-models", "diffusion": "generative-models",
    "gan": "generative-models", "vae": "generative-models",
    "multimodal": "multimodal",
    "rl": "reinforcement-learning", "ppo": "reinforcement-learning",
    "alignment": "alignment-rlhf", "dpo": "alignment-rlhf", "rlhf": "alignment-rlhf",
    "reasoning": "reasoning-agents", "agents": "reasoning-agents", "retrieval": "reasoning-agents",
    "rag": "reasoning-agents", "tools": "reasoning-agents", "prompting": "reasoning-agents",
    "quantization": "efficiency-systems", "peft": "efficiency-systems", "lora": "efficiency-systems",
    "finetuning": "efficiency-systems", "inference": "efficiency-systems", "serving": "efficiency-systems",
    "interpretability": "interpretability", "evaluation": "evaluation-trust",
    "gnn": "neural-network-foundations",
}
# free-form value -> technique
TOPIC2TECH = {
    "attention": "attention", "transformer": "transformer", "cnn": "cnn", "rnn": "rnn-lstm",
    "embeddings": "embeddings", "diffusion": "diffusion", "gan": "gan", "vae": "vae",
    "ssm": "ssm", "moe": "moe", "lora": "lora-peft", "peft": "lora-peft",
    "quantization": "quantization", "ppo": "ppo", "dpo": "dpo", "rlhf": "rlhf",
    "flashattention": "flashattention", "mlp": "mlp", "rag": "rag",
}
TOPIC2TASK_EXTRA = {"gnn": "graph"}

# --- youtube course -> (base topics, level, fallback topic) ------------------
# Focused courses get a base tag on every lecture; broad survey courses get no
# base (keyword-driven) plus a fallback used only when no keyword matches.
COURSE = {
    "3blue1brown-neural-networks": ([], "intro", "neural-network-foundations"),
    "andrej-karpathy-channel": ([], "intermediate", "language-models"),
    "andrej-karpathy-neural-networks-zero-to-": ([], "intermediate", "neural-network-foundations"),
    "deeplearningai": ([], "frontier", "ml-engineering"),
    "jeremy-howard-fastai-practical-deep-lear": ([], "intermediate", "neural-network-foundations"),
    "mit-6s191-introduction-to-deep-learning-": ([], "intro", "neural-network-foundations"),
    "stanford-cs224n-nlp-with-deep-learning-w": ([], "intermediate", "sequence-models-rnn"),
    "stanford-cs229-machine-learning-andrew-n": (["classical-ml"], "intermediate", "classical-ml"),
    "stanford-cs230-deep-learning-andrew-ng-2": (["neural-network-foundations"], "intermediate", "neural-network-foundations"),
    "stanford-cs231n-cnns-for-visual-recognit": (["computer-vision"], "intermediate", "computer-vision"),
    "stanford-cs236-deep-generative-models": (["generative-models"], "advanced", "generative-models"),
    "stanford-cs25-transformers-united": (["transformers-attention"], "advanced", "transformers-attention"),
    "stanford-cs336-language-modeling-from-sc": (["language-models"], "advanced", "language-models"),
    "yannic-kilcher-channel": ([], "frontier", "ai-industry-news"),
}

# --- title keyword -> topic (accumulate ALL matches) -------------------------
TOPIC_KW = [
    (["convolutional", "cnn", "resnet", "vgg", "densenet", "imagenet", "googlenet", "inception",
      "object detection", "segmentation", "yolo", "r-cnn", "vision transformer", "image classification",
      "visual recognition", "detr", "image recognition"], "computer-vision"),
    (["rnn", "lstm", "gru", "recurrent", "seq2seq", "sequence to sequence", "word2vec", "word vector",
      "glove", "word embedding", "dependency parsing", "coreference", "named entity",
      "machine translation", "tokeniz", "wavenet"], "sequence-models-rnn"),
    (["transformer", "attention", "positional encoding", "rope", "self-attention", "self attention"],
     "transformers-attention"),
    (["language model", "llm", " gpt", "gpt-", "gpt ", "bert", " t5", "llama", "mistral", "mixtral",
      "pretrain", "scaling law", "deepseek", "gemma", "qwen", "chinchilla"], "language-models"),
    (["mamba", "state space", "state-space", "rwkv", "xlstm", "retnet", "linformer", "performer",
      "longformer", "reformer", "flashattention", "flash attention", "linear attention",
      "sparse attention", "long context", "long-context", "infini-attention", "linear-time"],
     "efficient-architectures"),
    (["diffusion", "stable diffusion", "gan", "generative adversarial", " vae", "autoencoder",
      "variational", "score-based", "score based", "normalizing flow", "flow matching",
      "text-to-image", "dall", "image generation", "generative model", "lumiere"], "generative-models"),
    (["clip", "multimodal", "vision-language", "vision language", "speech", "audio", "asr",
      "text-to-speech", "perceiver", "v-jepa"], "multimodal"),
    (["reinforcement learning", "q-learning", "q learning", "policy gradient", "markov decision",
      "mdp", "alphago", "alphazero", "alphatensor", "alphageometry", "imitation learning",
      "decision transformer", "robot"], "reinforcement-learning"),
    (["rlhf", "instructgpt", "dpo", "orpo", "grpo", "preference", "alignment", "instruction tun",
      "red team", "constitutional", "reward model", "post-training", "post training"], "alignment-rlhf"),
    (["chain of thought", "chain-of-thought", "tree of thoughts", "self-consistency", "reasoning",
      "react", "toolformer", "agent", "agentic", "retrieval", " rag", "rag ", "test-time", "tool use"],
     "reasoning-agents"),
    (["quantization", "gptq", "awq", "int8", "lora", "qlora", "dora", "peft", "parameter-efficient",
      "cuda", "triton", " gpu", "kernel", "parallelism", "vllm", "paged attention", "kv cache",
      "kv-cache", "speculative decoding", "inference", "serving", "distillation"], "efficiency-systems"),
    (["interpretab", "circuits", "induction head", "superposition", "feature visualization",
      "probing", "rome", "mechanistic", "knowledge editing"], "interpretability"),
    (["benchmark", "evaluation", "mmlu", "humaneval", "big-bench", "calibration", "uncertainty",
      "adversarial", "robust", "fairness", "privacy", "hallucination", "trustworth"], "evaluation-trust"),
    (["svm", "support vector", "kernel", "decision tree", "ensemble", "naive bayes",
      "linear regression", "logistic regression", "pca", "expectation-max", "expectation max",
      "learning theory", "cross-validation", "gaussian discriminant", "factor analysis"], "classical-ml"),
    (["backprop", "gradient descent", "what is a neural network", "introduction to neural network",
      "training neural network", "introduction to deep learning", "intro to deep learning",
      "multilayer perceptron", "micrograd", "makemore", "batch norm", "batchnorm", "layer norm",
      "dropout", "regulariz", "activation function", "initializ", "improving neural"],
     "neural-network-foundations"),
    (["fasthtml", "nbdev", "mojo", "htmx", "solveit", "monstreui", "monsterui", "prompt engineering",
      "vibe coding", "production grade", "production-grade", "data stack", "live coding",
      "hackers guide", "apl", "ai dev", "agent engineering"], "ml-engineering"),
]
# strong "this is news/meta, not a subfield" markers -> ai-industry-news only
NEWS_KW = ["[ml news]", "ml news", "poster session", "vendor hall", "@ neurips", "neurips 2023",
           "sam altman", "is not coming", "not sentient", "got hacked", "fired", "ai licenses",
           "model licenses", "this week", "mansplainer", "hot waters", "changes the game"]

TOPIC_TASK = {
    "computer-vision": "vision", "sequence-models-rnn": "language", "language-models": "language",
    "transformers-attention": "language", "alignment-rlhf": "language", "reasoning-agents": "language",
    "multimodal": "multimodal", "reinforcement-learning": "rl-control", "classical-ml": "tabular-classical",
    "efficient-architectures": "language", "efficiency-systems": "general", "interpretability": "language",
    "evaluation-trust": "general", "generative-models": "general", "neural-network-foundations": "general",
    "ml-engineering": "general", "ai-industry-news": "general",
}
TECH_KW = {
    "cnn": ["convolutional", "resnet", "vgg", "densenet", "inception", "googlenet"],
    "transformer": ["transformer"], "attention": ["attention"],
    "diffusion": ["diffusion", "ddpm", "ddim", "score-based", "score based"],
    "gan": ["gan", "generative adversarial"], "vae": ["vae", "autoencoder", "variational"],
    "normalizing-flow": ["normalizing flow", "flow matching"],
    "ssm": ["mamba", "state space", "state-space", "rwkv", "xlstm", "retnet", " s4 "],
    "moe": ["mixture of experts", "mixture-of-experts", "mixtral", "switch transformer", " moe"],
    "lora-peft": ["lora", "qlora", "dora", "peft", "parameter-efficient", "adapter"],
    "quantization": ["quantization", "gptq", "awq", "int8", "int4"],
    "rlhf": ["rlhf", "instructgpt"], "dpo": ["dpo", "direct preference", "orpo"],
    "ppo": ["ppo", "proximal policy"],
    "cot": ["chain of thought", "chain-of-thought", "tree of thoughts", "self-consistency"],
    "rag": ["retrieval-augmented", " rag", "retrieval aug"],
    "flashattention": ["flashattention", "flash attention"],
    "embeddings": ["word2vec", "glove", "word vector", "word embedding"],
    "rnn-lstm": ["lstm", "gru", "recurrent neural"], "mlp": ["perceptron", "micrograd"],
}


def parse_fm(text):
    """Return (fm_lines, body_start_idx, fields) for a doc starting with ---."""
    lines = text.split("\n")
    if lines[0].strip() != "---":
        raise ValueError("no frontmatter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    fields = {}
    for ln in lines[1:end]:
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", ln)
        if m:
            fields[m.group(1)] = m.group(2)
    return lines, end, fields


def parse_list(val):
    val = val.strip()
    if not val or val == "[]":
        return []
    inner = val.strip("[]")
    return [x.strip().strip('"').strip("'") for x in inner.split(",") if x.strip()]


def unquote(val):
    return val.strip().strip('"').strip("'")


def kw_topics(title_l):
    found = []
    for subs, topic in TOPIC_KW:
        if any(s in title_l for s in subs) and topic not in found:
            found.append(topic)
    return found


def kw_techniques(title_l, existing_topics):
    techs = []
    for t in existing_topics:
        if t in TOPIC2TECH and TOPIC2TECH[t] not in techs:
            techs.append(TOPIC2TECH[t])
    for tech, subs in TECH_KW.items():
        if any(s in title_l for s in subs) and tech not in techs:
            techs.append(tech)
    return techs


def main():
    files = sorted(glob.glob(os.path.join(CORPUS, "**", "*.md"), recursive=True))
    files = [f for f in files if os.path.basename(f) != "INDEX.md"]
    manifest = defaultdict(list)
    short_alias_counts = defaultdict(int)
    docs = []

    # pass 1: compute everything, collect short-alias collisions
    for path in files:
        rel = os.path.relpath(path, ROOT)
        parts = rel.split(os.sep)
        category = parts[1]               # papers / youtube / web
        group = parts[2]                  # course / domain / filename
        text = open(path, encoding="utf-8").read()
        _, _, fm = parse_fm(text)
        title = unquote(fm.get("title", os.path.basename(path)))
        title_l = title.lower()
        existing = parse_list(fm.get("topics", "[]"))
        year = None
        if fm.get("published"):
            m = re.search(r"(\d{4})", fm["published"])
            if m:
                year = int(m.group(1))

        medium = {"papers": "paper", "youtube": "lecture", "web": "article"}[category]

        course = COURSE.get(group, ([], "intermediate", "neural-network-foundations"))

        # topics
        topics = []
        is_news = any(s in title_l for s in NEWS_KW)
        if medium == "lecture":
            topics += course[0]          # base tags (focused courses only)
        from_existing = False
        for t in existing:
            mapped = TOPIC_MAP.get(t)
            if mapped and mapped not in topics:
                topics.append(mapped); from_existing = True
        kw_hits = kw_topics(title_l)
        for t in kw_hits:
            if t not in topics:
                topics.append(t)
        # confidence: low if a lecture with no existing topics, no keyword hit,
        # and no focused-course base; or any doc in the two known-messy courses.
        messy = group in ("yannic-kilcher-channel", "deeplearningai")
        confident = bool(from_existing or kw_hits or (medium == "lecture" and course[0]))
        confidence = "low" if (messy or not confident) else "high"
        if is_news:
            topics = ["ai-industry-news"]
        if not topics:
            # fallback by course / medium when nothing matched
            if medium == "lecture":
                topics = [course[2]]
            elif medium == "article":
                topics = ["neural-network-foundations"]
            else:
                topics = ["language-models"]
        topics = topics[:4]

        # level
        if medium == "paper":
            level = "frontier" if (year and year >= 2023) else "advanced"
        elif medium == "article":
            level = "advanced" if "interpretability" in topics else "intermediate"
        else:
            level = course[1]

        # task-domain
        tasks = []
        for t in topics:
            task = TOPIC_TASK.get(t)
            if task and task not in tasks:
                tasks.append(task)
        for t in existing:
            if t in TOPIC2TASK_EXTRA and TOPIC2TASK_EXTRA[t] not in tasks:
                tasks.append(TOPIC2TASK_EXTRA[t])
        tasks = tasks[:2] or ["general"]

        techniques = kw_techniques(title_l, existing)[:5]

        # apply override (from reading the actual content) if present
        ov = OVERRIDES.get(rel)
        if ov:
            if ov.get("topics"):
                topics = ov["topics"][:4]
            if ov.get("level"):
                level = ov["level"]
            if ov.get("techniques"):
                techniques = ov["techniques"][:5]
            confidence = "high"
            tasks = []
            for t in topics:
                task = TOPIC_TASK.get(t)
                if task and task not in tasks:
                    tasks.append(task)
            tasks = tasks[:2] or ["general"]

        # aliases
        aliases = [title]
        short = title.split(":")[0].strip()
        if short and short != title and len(short) >= 4 and "|" not in short:
            short_alias_counts[short.lower()] += 1
            aliases.append(short)

        docs.append(dict(path=path, rel=rel, category=category, group=group, title=title,
                         medium=medium, topics=topics, level=level, tasks=tasks,
                         techniques=techniques, aliases=aliases, year=year,
                         confidence=confidence,
                         base=os.path.splitext(os.path.basename(path))[0]))

    # pass 2: build tags, drop colliding short aliases, write
    written = 0
    for d in docs:
        aliases = [d["aliases"][0]]
        if len(d["aliases"]) > 1 and short_alias_counts[d["aliases"][1].lower()] == 1:
            aliases.append(d["aliases"][1])

        tags = [f"topic/{t}" for t in d["topics"]]
        tags += [f"level/{d['level']}", f"medium/{d['medium']}"]
        tags += [f"task/{t}" for t in d["tasks"]]
        tags += [f"technique/{t}" for t in d["techniques"]]

        text = open(d["path"], encoding="utf-8").read()
        lines = text.split("\n")
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        fm_body = [ln for ln in lines[1:end]
                   if not re.match(r"^(tags|aliases):", ln)]
        new_fm = (["---"] + fm_body
                  + [f"aliases: {json.dumps(aliases, ensure_ascii=False)}",
                     f"tags: [{', '.join(tags)}]", "---"])
        new_text = "\n".join(new_fm + lines[end + 1:])
        open(d["path"], "w", encoding="utf-8").write(new_text)
        written += 1

        for t in d["topics"]:
            manifest[t].append(dict(base=d["base"], rel=d["rel"], title=d["title"],
                                    medium=d["medium"], level=d["level"], year=d["year"]))

    json.dump(manifest, open(os.path.join(os.path.dirname(__file__), ".tags_manifest.json"), "w"),
              indent=1, ensure_ascii=False)

    ambiguous = [dict(rel=d["rel"], title=d["title"], group=d["group"],
                      auto_topics=d["topics"]) for d in docs if d["confidence"] == "low"]
    json.dump(ambiguous, open(os.path.join(os.path.dirname(__file__), ".ambiguous.json"), "w"),
              indent=1, ensure_ascii=False)

    print(f"tagged {written} docs")
    print("docs per topic:")
    for t in TOPICS:
        print(f"  {len(manifest[t]):3d}  {t}")
    print("untagged:", len([d for d in docs if not d["topics"]]))
    print(f"low-confidence (to reclassify by reading): {len(ambiguous)}")
    from collections import Counter
    print("  by course:", dict(Counter(a["group"] for a in ambiguous)))


if __name__ == "__main__":
    sys.exit(main())
