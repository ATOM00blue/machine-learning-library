---
title: "Efficient & Long-Context Architectures"
aliases: ["Efficient & Long-Context Architectures"]
cssclasses: [moc]
---

# Efficient & Long-Context Architectures

> Alternatives to dense quadratic attention: sparse/linear attention, FlashAttention, long-context methods, and state-space models (Mamba, RWKV).

*205 documents.* 

## Start here

1. [MIT 6.S191 (2022): Deep Learning New Frontiers](../../corpus/youtube/mit-6s191-introduction-to-deep-learning-/wySXLRTxAGQ.md)  · 🎓 lecture · intro
2. [MIT 6.S191 (2023): The Modern Era of Statistics](../../corpus/youtube/mit-6s191-introduction-to-deep-learning-/p1NpGC8K-vs.md)  · 🎓 lecture · intermediate
3. [Fast Transformer Decoding: One Write-Head is All You Need](../../corpus/papers/1911.02150.md)  · 📄 paper · advanced
4. [Reformer: The Efficient Transformer](../../corpus/papers/2001.04451.md)  · 📄 paper · advanced
5. [Longformer: The Long-Document Transformer](../../corpus/papers/2004.05150.md)  · 📄 paper · advanced
6. [Rethinking Attention with Performers](../../corpus/papers/2009.14794.md)  · 📄 paper · advanced

## All documents

```dataview
TABLE WITHOUT ID
  link(file.link, default(title, file.name)) AS Document,
  default(source, "") AS Type,
  default(published, "") AS Date
FROM #topic/efficient-architectures and -"atlas"
SORT level ASC, published ASC
```

_(The list above renders in Obsidian with the Dataview plugin. On GitHub, browse **Start here** or the [full index](../../corpus/INDEX.md).)_

## Related topics

[Transformers & Attention](transformers-attention.md) · [Efficiency, Systems & Serving](efficiency-systems.md)

---

[← Atlas home](../Home.md)
