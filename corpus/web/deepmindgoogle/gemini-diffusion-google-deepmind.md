---
title: "Gemini Diffusion — Google DeepMind"
source: "web"
url: "https://deepmind.google/models/gemini-diffusion/"
domain: "deepmind.google"
name: "Gemini Diffusion — Google DeepMind"
published: "2025-05"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["language-models", "generative-models", "efficient-architectures"]
aliases: ["Gemini Diffusion — Google DeepMind"]
tags: [topic/language-models, topic/generative-models, topic/efficient-architectures, level/frontier, medium/article, task/language, task/general, technique/diffusion, technique/transformer]
---
## Summary

Gemini Diffusion is Google DeepMind's experimental text generation model that applies diffusion techniques to language generation instead of the conventional autoregressive (next-token prediction) paradigm. Rather than producing tokens one at a time from left to right, the model "learns to generate outputs by refining noise, step-by-step," enabling a fundamentally different generation process that iteratively denoises toward coherent text.

A key architectural distinction is that Gemini Diffusion produces entire blocks of tokens simultaneously rather than sequentially, which the team claims improves coherence across the output and allows the model to correct errors during the refinement process rather than being locked into earlier decisions. This also unlocks substantially faster inference: the model reaches 1,479 tokens per second, described as "significantly faster" than comparable autoregressive models.

On coding benchmarks the model scores 89.6% on HumanEval and 76.0% on MBPP, demonstrating competitive performance with larger models, though it trails on some reasoning benchmarks. Google DeepMind has released it as an experimental demonstration for developer exploration, signaling early-stage but serious investment in diffusion as an alternative generation paradigm for large language models.

## Key points

- Diffusion-based LLM that refines noise iteratively rather than generating tokens autoregressively
- Generates entire token blocks simultaneously, improving coherence and enabling mid-generation error correction
- Achieves 1,479 tokens per second — significantly faster than comparable autoregressive models
- Competitive coding benchmark results: 89.6% HumanEval, 76.0% MBPP
- Released as an experimental model by Google DeepMind for developer testing

---
*Source: https://deepmind.google/models/gemini-diffusion/*
