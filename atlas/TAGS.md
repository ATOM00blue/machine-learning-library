# Tag vocabulary

Every doc carries a controlled `tags:` list (layered on top of the original free-form `topics:` field, which is left untouched). Five namespaces:

## `topic/` — subject (17, multi-valued)

| Tag | Docs | Covers |
|---|--:|---|
| `topic/neural-network-foundations` | 84 | MLPs, the forward pass, backpropagation, and the training machinery — optimizers (SGD/Adam), normalization, dropout, initialization. The entry point. |
| `topic/classical-ml` | 35 | Non-neural methods: regression, SVMs/kernels, trees & ensembles, naive Bayes, EM, PCA, and learning theory. |
| `topic/computer-vision` | 89 | CNNs (ResNet, VGG, DenseNet), image classification, object detection & segmentation, and vision transformers (ViT, DETR, MAE). |
| `topic/sequence-models-rnn` | 76 | Word embeddings, RNN/LSTM/GRU, seq2seq with attention, neural machine translation, and classic NLP tasks. The bridge to transformers. |
| `topic/transformers-attention` | 140 | The transformer architecture and attention itself: self-/multi-head attention, positional encodings, and architecture explainers. |
| `topic/language-models` | 377 | Building LLMs end-to-end: pretraining, the BERT/GPT/T5/LLaMA model reports, scaling laws, emergent abilities, and mixture-of-experts. |
| `topic/efficient-architectures` | 205 | Alternatives to dense quadratic attention: sparse/linear attention, FlashAttention, long-context methods, and state-space models (Mamba, RWKV). |
| `topic/generative-models` | 144 | Deep generative modeling: VAEs, GANs, normalizing flows, autoregressive models, and diffusion incl. latent/stable diffusion and text-to-image. |
| `topic/multimodal` | 69 | Models bridging modalities: contrastive vision-language (CLIP), text-to-image, audio/speech transformers, and general multimodal work. |
| `topic/reinforcement-learning` | 45 | RL as a subfield: MDPs, value/policy iteration, Q-learning, policy gradients, deep RL, imitation, and game-playing agents. |
| `topic/alignment-rlhf` | 93 | Post-training LLMs to human preferences: instruction tuning/SFT, RLHF/InstructGPT, reward modeling, DPO/ORPO/GRPO, and safety alignment. |
| `topic/reasoning-agents` | 166 | Eliciting and structuring reasoning, and tool-using agents: chain-of-thought, tree-of-thoughts, test-time compute, ReAct/Toolformer, and RAG. |
| `topic/efficiency-systems` | 44 | Making models cheap to adapt, train, and serve: quantization, LoRA/PEFT, CUDA/Triton kernels, parallelism, and paged-attention serving (vLLM). |
| `topic/interpretability` | 29 | Reverse-engineering trained models: mechanistic interpretability (circuits, superposition, induction heads), probing, and knowledge editing. |
| `topic/evaluation-trust` | 82 | Measuring and stress-testing models: benchmarks, evaluation methodology, calibration/uncertainty, robustness, bias/fairness, and hallucination. |
| `topic/ml-engineering` | 109 | Hands-on build & ship: production agent engineering, app/agent frameworks, dev tooling (FastHTML, nbdev, CUDA-for-Python), and prompt engineering. |
| `topic/ai-industry-news` | 71 | News, commentary, and the broader ecosystem: model-release roundups, policy, AGI debate, conference coverage, and applied 'AI for X' talks. |

## `level/` — difficulty

`level/intro` · `level/intermediate` · `level/advanced` · `level/frontier`

## `medium/` — format

`medium/paper` · `medium/lecture` · `medium/article`

## `task/` — modality / problem domain

`task/vision` · `task/language` · `task/speech-audio` · `task/multimodal` · `task/graph` · `task/rl-control` · `task/tabular-classical` · `task/general`

## `technique/` — architecture / method

`technique/mlp` · `technique/cnn` · `technique/rnn-lstm` · `technique/transformer` · `technique/attention` · `technique/diffusion` · `technique/gan` · `technique/vae` · `technique/normalizing-flow` · `technique/ssm` · `technique/moe` · `technique/lora-peft` · `technique/quantization` · `technique/rlhf` · `technique/dpo` · `technique/ppo` · `technique/cot` · `technique/rag` · `technique/flashattention` · `technique/embeddings`

---

Tags are auto-assigned (keyword rules + a content-reading pass); they're great for narrowing, but full-text search is the ground truth when a filter looks sparse.
