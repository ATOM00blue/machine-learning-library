---
title: "Alignment, RLHF & Preference Tuning"
aliases: ["Alignment, RLHF & Preference Tuning"]
cssclasses: [moc]
---

# Alignment, RLHF & Preference Tuning

> Post-training LLMs to human preferences: instruction tuning/SFT, RLHF/InstructGPT, reward modeling, DPO/ORPO/GRPO, and safety alignment.

*25 documents.* 

## Start here

1. [ChatGPT: This AI has a JAILBREAK?! (Unbelievable AI Progress)](../../corpus/youtube/yannic-kilcher-channel/0A8ljAkdFtg.md)  · 🎓 lecture · intro
2. [MIT 6.S191 (2025): A Hipocratic Oath, for *your* AI (Comet ML)](../../corpus/youtube/mit-6s191-introduction-to-deep-learning-/CyCUZAf8xSU.md)  · 🎓 lecture · intro
3. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](../../corpus/papers/2210.17323.md)  · 📄 paper · advanced
4. [Training language models to follow instructions with human feedback](../../corpus/papers/2203.02155.md)  · 📄 paper · advanced
5. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](../../corpus/papers/2305.18290.md)  · 📄 paper · frontier

## All documents

```dataview
TABLE WITHOUT ID
  link(file.link, default(title, file.name)) AS Document,
  default(source, "") AS Type,
  default(published, "") AS Date
FROM #topic/alignment-rlhf and -"atlas"
SORT level ASC, published ASC
```

_(The list above renders in Obsidian with the Dataview plugin. On GitHub, browse **Start here** or the [full index](../../corpus/INDEX.md).)_

## Related topics

[Language Models & Pretraining](language-models.md) · [Reinforcement Learning](reinforcement-learning.md) · [Reasoning & Agents](reasoning-agents.md)

---

[← Atlas home](../Home.md)
