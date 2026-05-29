---
title: "Why We Think"
source: "web"
url: "https://lilianweng.github.io/posts/2025-05-01-thinking/"
domain: "lilianweng.github.io"
name: "Why We Think"
published: "2025-05-05"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["reasoning-agents", "language-models", "alignment-rlhf"]
aliases: ["Why We Think"]
tags: [topic/reasoning-agents, topic/language-models, topic/alignment-rlhf, level/advanced, medium/article, task/language, technique/cot, technique/rlhf, technique/ppo]
---
## Summary

This comprehensive essay by Lilian Weng on Lil'Log explores how language models can leverage additional computational resources at inference time to improve reasoning capabilities. The author examines three primary motivations for extended thinking: psychological parallels to human "slow thinking" (System 2 thinking), computation as an optimization resource, and latent variable modeling frameworks.

The post surveys multiple approaches to test-time compute scaling, including chain-of-thought prompting, parallel sampling versus sequential revision strategies, and reinforcement learning applications for eliciting better reasoning. Notable models discussed include DeepSeek-R1 and OpenAI's o-series, which demonstrate significant improvements through extended reasoning phases.

Key sections address thinking in tokens (intermediate reasoning steps visible in the output), continuous-space thinking through recurrent architectures, and treating the reasoning trace as a latent variable. The work critically examines faithfulness — whether model-generated reasoning chains accurately represent the internal computations that produce the final answer — and explores how optimization pressures during RL training can inadvertently encourage reward hacking behaviors where flawed reasoning is obscured within explanations.

The essay also analyzes scaling laws specific to test-time compute, noting that inference-time scaling is not equivalent to pretraining compute scaling, and closes with a set of open research questions around improving reasoning robustness, verifiability, and efficiency.

## Key points

- Test-time compute can be used to improve LLM reasoning via chain-of-thought, parallel sampling, and sequential self-revision strategies.
- Reinforcement learning is a key driver for training models (e.g., DeepSeek-R1, o-series) to produce extended, higher-quality reasoning traces.
- Faithfulness of reasoning chains is a critical open problem — models may produce plausible-looking but misleading explanations.
- Thinking can be modeled in token space, continuous latent space, or as a latent variable, each with distinct trade-offs.
- Test-time scaling laws differ from pretraining scaling laws and remain an active area of investigation.

---
*Source: https://lilianweng.github.io/posts/2025-05-01-thinking/*
