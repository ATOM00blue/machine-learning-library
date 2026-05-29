---
title: "TimesFM 2.5: Smaller, Longer-Context Foundation Model Leading GIFT-Eval"
source: "web"
url: "https://huggingface.co/google/timesfm-2.5-200m-pytorch"
domain: "huggingface.co"
name: "TimesFM 2.5: Smaller, Longer-Context Foundation Model Leading GIFT-Eval"
published: "2025-09-15"
fetched_at: "2026-05-29T00:00:00Z"
topics: ["sequence-models-rnn", "efficient-architectures", "language-models"]
aliases: ["TimesFM 2.5: Smaller, Longer-Context Foundation Model Leading GIFT-Eval", "TimesFM 2.5"]
tags: [topic/sequence-models-rnn, topic/efficient-architectures, topic/language-models, level/frontier, medium/article, task/language, technique/transformer, technique/attention]
---
## Summary

TimesFM 2.5 is a pretrained decoder-only foundation model developed by Google Research for general-purpose time-series forecasting. This release represents a significant compression and capability expansion over prior versions: the model size was reduced from 500M to 200M parameters, while the supported context length was extended from 2,048 to 16,384 time steps. The forecasting horizon is configurable up to 256 steps by default, with an optional continuous quantile head extending to 1,000-step horizons.

The model supports both point forecasts and quantile-based uncertainty estimation across multiple percentile levels (10th–90th). It was pretrained on a diverse corpus including the GiftEvalPretrain dataset, Wikimedia pageviews, Google Trends queries, and synthetic data, enabling strong zero-shot generalization across time-series domains.

Key architectural improvements in 2.5 include the elimination of the frequency indicator requirement (simplifying inference), fused QKV matrices for speed, and new forecasting flags for normalized inputs and flip invariance. The model achieves leading results on the GIFT-Eval benchmark. It is distributed as a PyTorch checkpoint and supports torch.compile for production deployment.

TimesFM 2.5 builds on the original ICML 2024 paper "A decoder-only foundation model for time-series forecasting" and demonstrates that a smaller, more efficient foundation model can outperform larger predecessors when trained on curated, diverse temporal data.

## Key points

- Reduced from 500M to 200M parameters while expanding context from 2,048 to 16,384 time steps
- Achieves leading performance on the GIFT-Eval time-series forecasting benchmark
- Supports zero-shot forecasting with point estimates and quantile uncertainty outputs
- Eliminates frequency indicator requirement, simplifying deployment
- Released by Google Research in September 2025 as an open model checkpoint on Hugging Face

---
*Source: https://huggingface.co/google/timesfm-2.5-200m-pytorch*
