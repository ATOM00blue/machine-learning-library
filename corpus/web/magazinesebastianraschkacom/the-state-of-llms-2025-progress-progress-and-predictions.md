---
title: "The State Of LLMs 2025: Progress, Progress, and Predictions"
source: "web"
url: "https://magazine.sebastianraschka.com/p/state-of-llms-2025"
domain: "magazine.sebastianraschka.com"
name: "The State Of LLMs 2025: Progress, Progress, and Predictions"
published: "2025-12-30"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["language-models", "alignment-rlhf", "reasoning-agents"]
aliases: ["The State Of LLMs 2025: Progress, Progress, and Predictions", "The State Of LLMs 2025"]
tags: [topic/language-models, topic/alignment-rlhf, topic/reasoning-agents, level/advanced, medium/article, task/language, technique/rlhf, technique/ppo, technique/dpo, technique/lora-peft, technique/cot]
---
## Summary

Sebastian Raschka's end-of-year review covers the major developments in large language models throughout 2025. The central theme is the rise of reasoning models powered by Reinforcement Learning with Verifiable Rewards (RLVR) and the GRPO algorithm, spurred by DeepSeek R1's release in early 2025. DeepSeek R1 attracted attention for being an open-weight model competitive with top proprietary systems, and for dramatically lowering perceived training cost estimates — the DeepSeek V3 671B model was reportedly trained for roughly $5 million in compute credits, an order of magnitude below prior assumptions.

Raschka organizes LLM development focus by year: 2022 was RLHF + PPO, 2023 saw LoRA and parameter-efficient fine-tuning, 2024 focused on mid-training with synthetic data and domain-specific stages, and 2025 was dominated by RLVR + GRPO. Academic researchers embraced GRPO because it is conceptually tractable and experimentally feasible at moderate scale, leading to a wave of GRPO refinements (e.g., DAPO, Dr. GRPO) that were adopted in production pipelines.

The article also addresses inference-time scaling (self-consistency, self-refinement), benchmark saturation problems, architectural trends such as hybrid and MoE models, the emergence of new open-weight leaders like Qwen3 displacing Llama, the standardization of MCP for tool use, and the landscape of multi-modal and agentic systems. The piece closes with 2026 predictions, including diffusion LLMs, RLVR expanding beyond math and code, and inference-time improvements outpacing training-side gains.

## Key points

- RLVR + GRPO emerged as the defining post-training paradigm of 2025, replacing RLHF + PPO as the dominant focus, enabling reasoning models that learn from verifiable correctness signals without expensive human labels.
- DeepSeek R1 and V3 reset cost assumptions for frontier model training, with estimates an order of magnitude lower than previously believed.
- Inference-time scaling methods (self-consistency, self-refinement, extended chain-of-thought) became a major lever for performance gains alongside training improvements.
- The open-weight ecosystem shifted: Qwen3 and DeepSeek overtook Llama in downloads and derivative models, and MCP became the de facto standard for tool and data access in agentic LLM pipelines.
- 2026 predictions: RLVR expanding to chemistry/biology, classical RAG fading in favor of long-context models, inference-time improvements dominating apparent progress over pure training advances.

---
*Source: https://magazine.sebastianraschka.com/p/state-of-llms-2025*
