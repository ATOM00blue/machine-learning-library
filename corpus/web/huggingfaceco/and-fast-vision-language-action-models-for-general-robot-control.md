---
title: "π₀ and π₀-FAST: Vision-Language-Action Models for General Robot Control"
source: "web"
url: "https://huggingface.co/blog/pi0"
domain: "huggingface.co"
name: "π₀ and π₀-FAST: Vision-Language-Action Models for General Robot Control"
published: "2025-02-04"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["multimodal", "reinforcement-learning", "efficient-architectures"]
aliases: ["π₀ and π₀-FAST: Vision-Language-Action Models for General Robot Control", "π₀ and π₀-FAST"]
tags: [topic/multimodal, topic/reinforcement-learning, topic/efficient-architectures, level/frontier, medium/article, task/multimodal, task/rl-control, technique/transformer, technique/attention, technique/embeddings]
---
## Summary

This Hugging Face blog post introduces π0 (pi-zero) and π0-FAST, robotics foundation models developed by Physical Intelligence and ported to the Hugging Face LeRobot library. π0 is a Vision-Language-Action (VLA) model trained on 7 robotic platforms and 68 tasks, employing flow matching to generate smooth real-time action trajectories at 50Hz. Unlike standard Vision-Language Models (VLMs) that process images and text, VLAs extend this paradigm with action and observation state tokens that allow the model to output motor commands.

The post provides a detailed walkthrough of the attention mechanisms used in robotic policies: state tokens representing the current environment configuration, action tokens encoding motor commands, and prefix tokens capturing scene context. Careful masking of cross-token attention is critical to enable efficient and coherent multi-modal reasoning.

π0-FAST extends π0 by introducing FAST (Frequency-space Action Sequence Tokenization), which applies the Discrete Cosine Transform to compress action sequences into compact discrete tokens. This enables 5x faster training compared to the base π0 model while retaining high action quality. The implementation leverages PyTorch's FlexAttention API to handle the complex 2D attention masks required by this multi-modal architecture efficiently.

Both models are released as part of LeRobot, aiming to enable generalist robot intelligence that transfers across diverse robot embodiments and task domains.

## Key points

- π0 is a VLA model trained across 7 robot platforms and 68 tasks, producing smooth 50Hz action trajectories via flow matching.
- VLAs augment VLMs with action and state tokens, enabling direct motor command generation from vision and language inputs.
- π0-FAST introduces Discrete Cosine Transform-based tokenization of action sequences, achieving 5x faster training.
- PyTorch FlexAttention is used to efficiently compute the complex 2D attention masks required across heterogeneous token types.
- Both models are open-sourced via Hugging Face LeRobot for the broader robotics research community.

---
*Source: https://huggingface.co/blog/pi0*
