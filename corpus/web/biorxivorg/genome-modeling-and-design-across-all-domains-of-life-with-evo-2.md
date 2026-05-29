---
title: "Genome modeling and design across all domains of life with Evo 2"
source: "web"
url: "https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1"
domain: "biorxiv.org"
name: "Genome modeling and design across all domains of life with Evo 2"
published: "2025-02-21"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["language-models", "generative-models", "efficient-architectures"]
aliases: ["Genome modeling and design across all domains of life with Evo 2"]
tags: [topic/language-models, topic/generative-models, topic/efficient-architectures, level/frontier, medium/article, task/language, task/general, technique/transformer, technique/ssm, technique/embeddings]
---
## Summary

Evo 2 is a biological foundation model trained on 9.3 trillion DNA base pairs from a highly curated genomic atlas (OpenGenome2) spanning all domains of life — bacteria, archaea, and eukaryotes. The model is released at both 7B and 40B parameter scales and features an unprecedented 1 million token context window at single-nucleotide resolution, enabling it to reason over entire genomic regions in a single forward pass.

Trained purely on DNA sequence, Evo 2 learns to accurately predict the functional impacts of genetic variants — including noncoding pathogenic mutations and clinically significant BRCA1 variants — without any task-specific fine-tuning. Mechanistic interpretability analyses show that the model autonomously learns biologically meaningful features such as exon–intron boundaries, transcription factor binding sites, protein structural elements, and prophage genomic regions.

On the generative side, Evo 2 can produce mitochondrial, prokaryotic, and eukaryotic sequences at genome scale with greater naturalness and coherence than prior methods. Inference-time search enables controllable generation of epigenomic structure, with the paper presenting the first inference-time scaling results in biology.

The authors fully open-source Evo 2, releasing model weights, training code, inference code, and the OpenGenome2 dataset to enable broad exploration and design of biological complexity. This work represents a significant step toward building truly general genomic foundation models capable of understanding and engineering life across all domains.

## Key points

- 40B-parameter genomic foundation model trained on 9.3T DNA base pairs with 1M-token context at single-nucleotide resolution
- Zero-shot prediction of functional effects of genetic variants including clinically significant BRCA1 mutations
- Mechanistic interpretability reveals emergent learning of biological features (splice sites, TF binding sites, protein structure)
- Genome-scale generative capabilities across mitochondrial, prokaryotic, and eukaryotic sequences; first inference-time scaling results in biology
- Fully open release: model weights, training/inference code, and OpenGenome2 dataset

---
*Source: https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1*
