---
title: "Efficiency, Systems & Serving"
aliases: ["Efficiency, Systems & Serving"]
cssclasses: [moc]
---

# Efficiency, Systems & Serving

> Making models cheap to adapt, train, and serve: quantization, LoRA/PEFT, CUDA/Triton kernels, parallelism, and paged-attention serving (vLLM).

*44 documents.* 

## Start here

1. [AI Dev 26 x SF | Manos Koukoumidis & Stefan Webb: VibeML: Build your AI model in hours, not months](../../corpus/youtube/deeplearningai/MlVuNCP9uxs.md)  · 🎓 lecture · intermediate
2. [Getting Started With CUDA for Python Programmers](../../corpus/youtube/jeremy-howard-fastai-practical-deep-lear/nOxKexn3iBo.md)  · 🎓 lecture · intermediate
3. [LoRA: Low-Rank Adaptation of Large Language Models](../../corpus/papers/2106.09685.md)  · 📄 paper · advanced
4. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../../corpus/papers/2210.17323.md)  · 📄 paper · advanced
5. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](../../corpus/papers/2208.07339.md)  · 📄 paper · advanced
6. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](../../corpus/papers/2306.00978.md)  · 📄 paper · frontier

## All documents

```dataview
TABLE WITHOUT ID
  link(file.link, default(title, file.name)) AS Document,
  default(source, "") AS Type,
  default(published, "") AS Date
FROM #topic/efficiency-systems and -"atlas"
SORT level ASC, published ASC
```

_(The list above renders in Obsidian with the Dataview plugin. On GitHub, browse **Start here** or the [full index](../../corpus/INDEX.md).)_

## Related topics

[Language Models & Pretraining](language-models.md) · [Efficient & Long-Context Architectures](efficient-architectures.md)

---

[← Atlas home](../Home.md)
