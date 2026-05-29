---
title: "Chai-1: Decoding the molecular interactions of life"
source: "web"
url: "https://www.biorxiv.org/content/10.1101/2024.10.10.615955v1"
domain: "biorxiv.org"
name: "Chai-1: Decoding the molecular interactions of life"
published: "2024-10-11"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["computer-vision", "efficient-architectures", "generative-models"]
aliases: ["Chai-1: Decoding the molecular interactions of life", "Chai-1"]
tags: [topic/computer-vision, topic/efficient-architectures, topic/generative-models, level/frontier, medium/article, task/vision, task/language, technique/transformer, technique/attention, technique/embeddings]
---
## Summary

Chai-1 is a multi-modal foundation model for molecular structure prediction that achieves state-of-the-art performance across a variety of tasks relevant to drug discovery. The model unifies prediction of diverse biomolecular entities — proteins, small molecules, DNA, RNA, glycosylations, and more — within a single architecture.

A key distinguishing feature of Chai-1 is its support for experimental restraints: users can optionally provide wet-lab-derived contact or covalent bond data to guide the structure prediction process, which boosts benchmark performance by double-digit percentage points. This makes Chai-1 particularly valuable for integrating computational predictions with experimental observations in real drug discovery pipelines.

Chai-1 also supports single-sequence inference without multiple sequence alignments (MSAs), preserving most of its performance and dramatically reducing computational and data requirements for users who lack pre-computed MSA databases. The model weights and inference code are released as a Python package for non-commercial use, and a free web interface is available for both academic and commercial drug discovery purposes under Apache 2.0 licensing.

The system represents a significant advance toward a general-purpose biomolecular structure predictor, building upon and extending the paradigm established by AlphaFold 2 and RoseTTAFold. It was released by Chai Discovery in October 2024 alongside a technical report on bioRxiv.

## Key points

- Multi-modal foundation model predicting structures of proteins, small molecules, DNA, RNA, and glycosylations in a unified framework
- Experimental restraint prompting boosts performance by double-digit percentage points over baseline
- Single-sequence mode (no MSAs required) retains most performance, lowering the barrier to use
- Released as open-weight Python package (non-commercial) and free web server (including commercial drug discovery)
- State-of-the-art across diverse drug discovery benchmarks at time of release

---
*Source: https://www.biorxiv.org/content/10.1101/2024.10.10.615955v1*
