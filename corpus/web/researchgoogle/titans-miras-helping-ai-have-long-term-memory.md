---
title: "Titans + MIRAS: Helping AI have long-term memory"
source: "web"
url: "https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/"
domain: "research.google"
name: "Titans + MIRAS: Helping AI have long-term memory"
published: "2025-12-04"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["sequence-models-rnn", "efficient-architectures", "language-models"]
aliases: ["Titans + MIRAS: Helping AI have long-term memory", "Titans + MIRAS"]
tags: [topic/sequence-models-rnn, topic/efficient-architectures, topic/language-models, level/frontier, medium/article, task/language, technique/rnn-lstm, technique/attention, technique/transformer]
---
## Summary

Google Research introduces two complementary innovations for advancing AI sequence modeling: Titans and MIRAS. Titans is a neural architecture that combines the speed of RNNs with the accuracy of Transformers by using deep neural networks as memory modules. Rather than compressing history into fixed-size recurrent states, Titans actively learns and updates its memory parameters during inference using a "surprise metric" — a gradient-based mechanism that detects significant discrepancies between expected and actual inputs, selectively deciding what to remember. Momentum and adaptive weight decay are incorporated to enable efficient long-term retention without unbounded parameter growth.

MIRAS (Memory as a Recursive Associative Structure) provides a unified theoretical framework that casts sequence modeling as an associative memory problem, organized around four design dimensions: memory architecture, attentional bias, retention gates, and optimization algorithms. Under this framework, three new model variants — YAAD, MONETA, and MEMORA — are derived and shown to outperform strong baselines including Mamba-2 and Transformer++ on language modeling and reasoning benchmarks. Critically, these models maintain linear inference complexity and support parallelizable training.

Most notably, Titans and MIRAS variants handle extreme long-context scenarios exceeding 2 million tokens, surpassing GPT-4 on specific benchmarks (BABILong) despite having fewer parameters.

## Key points

- Titans replaces fixed-size recurrent states with deep neural network memory modules that update via surprise-driven gradient signals at inference time
- MIRAS unifies sequence modeling architectures under a four-axis associative memory design space
- Models achieve linear scaling complexity while supporting context lengths beyond 2 million tokens
- Three new MIRAS variants (YAAD, MONETA, MEMORA) outperform Mamba-2 and Transformer++ on language and reasoning tasks

---
*Source: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/*
