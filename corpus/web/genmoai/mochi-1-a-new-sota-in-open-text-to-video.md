---
title: "Mochi 1: A new SOTA in open text-to-video"
source: "web"
url: "https://www.genmo.ai/blog/mochi-1-a-new-sota-in-open-text-to-video"
domain: "genmo.ai"
name: "Mochi 1: A new SOTA in open text-to-video"
published: "2025-09-10"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["generative-models", "multimodal", "efficient-architectures"]
aliases: ["Mochi 1: A new SOTA in open text-to-video", "Mochi 1"]
tags: [topic/generative-models, topic/multimodal, topic/efficient-architectures, level/frontier, medium/article, task/general, task/multimodal, technique/diffusion, technique/transformer, technique/attention, technique/embeddings, technique/vae]
---
## Summary

Mochi 1 is an open-source video generation model released by Genmo that achieves state-of-the-art results in text-to-video synthesis. The model generates videos at 480p resolution for up to 5.4 seconds at 30 frames per second, exhibiting high temporal coherence and realistic motion dynamics.

The architecture is built around a 10 billion parameter Asymmetric Diffusion Transformer (AsymmDiT), making it the largest openly released video generative model at the time of release. A custom video VAE compresses video content 96x, and a single T5-XXL language model handles prompt encoding rather than relying on multiple pretrained text encoders.

Key technical innovations include 3D rotary positional embeddings supporting context windows of up to 44,520 video tokens, and an asymmetric design in which the visual stream carries nearly four times more parameters than the text stream — enabling more expressive visual generation while keeping text conditioning efficient.

Mochi 1 demonstrates strong prompt adherence and smooth motion quality, significantly narrowing the gap between open and closed-source video generation systems. The model is released under the Apache 2.0 license and is freely available on HuggingFace and GitHub, with a free web playground also provided. The announcement was accompanied by news of $28.4 million in Series A funding for Genmo.

## Key points

- 10B parameter Asymmetric Diffusion Transformer (AsymmDiT) — largest openly released video generative model at release
- Custom video VAE achieving 96x spatial-temporal compression
- 3D rotary positional embeddings enabling 44,520-token video context windows
- Single T5-XXL text encoder for prompt conditioning; asymmetric visual/text parameter split
- Released under Apache 2.0; available on HuggingFace and GitHub

---
*Source: https://www.genmo.ai/blog/mochi-1-a-new-sota-in-open-text-to-video*
