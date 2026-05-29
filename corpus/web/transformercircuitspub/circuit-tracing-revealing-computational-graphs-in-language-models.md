---
title: "Circuit Tracing: Revealing Computational Graphs in Language Models"
source: "web"
url: "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"
domain: "transformer-circuits.pub"
name: "Circuit Tracing: Revealing Computational Graphs in Language Models"
published: "2025-03-27"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["interpretability", "language-models", "transformers-attention"]
aliases: ["Circuit Tracing: Revealing Computational Graphs in Language Models", "Circuit Tracing"]
tags: [topic/interpretability, topic/language-models, topic/transformers-attention, level/frontier, medium/article, task/language, technique/transformer, technique/embeddings, technique/attention]
---
## Summary

This paper introduces a methodology for uncovering the computational mechanisms underlying language model behaviors through "circuit tracing." The authors develop an interpretable replacement model by substituting cross-layer transcoders (CLTs) — sparse coding models — for multi-layer perceptrons, enabling detailed mechanistic analysis of how information flows and transforms inside a transformer. They construct "attribution graphs" depicting feature-to-feature interactions on specific prompts using linear approximations with frozen attention patterns and normalization statistics.

The approach includes visualization tooling, validation through perturbation experiments, and application to both a simpler 18-layer model and a frontier production model (Claude 3.5 Haiku). Case studies demonstrate the method on factual recall, multi-digit addition, and acronym generation tasks. Key contributions include showing that CLTs achieve approximately 50% output matching while providing substantially superior interpretability compared to raw neurons or per-layer transcoders; developing pruning algorithms to make large attribution graphs tractable for human analysis; and establishing quantitative evaluation metrics for both interpretability quality and mechanistic faithfulness.

The work is part of Anthropic's Transformer Circuits research thread and represents a significant methodological step toward scalable mechanistic interpretability of large language models, moving from toy or small models toward real-world frontier systems.

## Key points

- Introduces cross-layer transcoders (CLTs) as sparse, interpretable replacements for MLPs, enabling construction of prompt-level attribution graphs
- Attribution graphs capture feature-to-feature causal flow with linear approximations, making internal computation human-readable
- Method validated on a frontier model (Claude 3.5 Haiku) with case studies on factual recall, addition, and acronym tasks
- Pruning algorithms and quantitative faithfulness metrics make the approach scalable and evaluable beyond qualitative inspection

---
*Source: https://transformer-circuits.pub/2025/attribution-graphs/methods.html*
