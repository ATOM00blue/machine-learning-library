---
title: "On the Biology of a Large Language Model"
source: "web"
url: "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"
domain: "transformer-circuits.pub"
name: "On the Biology of a Large Language Model"
published: "2025-03-27"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["interpretability", "language-models", "reasoning-agents"]
aliases: ["On the Biology of a Large Language Model"]
tags: [topic/interpretability, topic/language-models, topic/reasoning-agents, level/frontier, medium/article, task/language, technique/transformer, technique/attention, technique/embeddings]
---
## Summary

This paper investigates Claude 3.5 Haiku's internal mechanisms using attribution graphs — a circuit tracing methodology that maps computational pathways within the model. Rather than treating the model as a black box, researchers build interpretable "replacement models" using cross-layer transcoders to identify features and trace their interactions.

The study examines diverse phenomena across multiple case studies. In multi-step reasoning, the model genuinely performs two-hop deduction (Dallas → Texas → Austin) alongside shortcut pathways. Poetry analysis reveals forward planning, where the model pre-selects rhyming words before composing lines. Multilingual circuits demonstrate both language-independent abstract operations and language-specific components, with English mechanistically privileged as a default. Arithmetic circuits show lookup-table features generalizing across contexts like astronomical data and financial tables.

Additional case studies analyze medical diagnosis (where internal representations guide symptom inquiries), hallucination mechanisms (contrasting "known answer" features that suppress refusals), refusal circuits (aggregating specific harms into general harmful-request features), and jailbreak vulnerabilities (where letter-stitching bypasses immediate recognition).

The researchers conclude that Claude employs sophisticated strategies including intermediate reasoning steps, forward and backward planning, and metacognitive awareness. A key finding is that the model's internal reasoning often diverges from its stated explanations, highlighting gaps between actual mechanisms and articulated reasoning. The authors note that attribution graphs provide satisfactory insight for approximately a quarter of tested prompts, representing a meaningful but bounded step toward full mechanistic understanding.

## Key points

- Attribution graphs (cross-layer transcoder-based replacement models) enable circuit-level tracing of Claude 3.5 Haiku's internal computations across diverse tasks.
- The model performs genuine multi-step deduction, forward rhyme planning in poetry, and uses a mix of language-specific and language-universal circuits.
- Hallucination and jailbreak mechanisms are characterized at the feature/circuit level, showing how familiar-entity suppression of refusals can be exploited.
- Internal reasoning frequently diverges from the model's own stated explanations, underscoring the limits of self-report as a mechanistic account.
- The method works well on roughly a quarter of tested prompts, indicating substantial room for future interpretability progress.

---
*Source: https://transformer-circuits.pub/2025/attribution-graphs/biology.html*
