---
title: "11. Attention Mechanisms and Transformers"
source: "web"
url: "https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html"
domain: "d2l.ai"
name: "d2l - Attention"
fetched_at: "2026-05-26T23:30:19Z"
topics: ["attention", "transformer"]
---

# 11. Attention Mechanisms and Transformers Â¶

The earliest years of the deep learning boom were driven primarily by
results produced using the multilayer perceptron, convolutional network,
and recurrent network architectures. Remarkably, the model architectures
that underpinned many of deep learningâs breakthroughs in the 2010s had
changed remarkably little relative to their antecedents despite the
lapse of nearly 30 years. While plenty of new methodological innovations
made their way into most practitionerâs toolkitsâReLU activations,
residual layers, batch normalization, dropout, and adaptive learning
rate schedules come to mindâthe core underlying architectures were
clearly recognizable as scaled-up implementations of classic ideas.
Despite thousands of papers proposing alternative ideas, models
resembling classical convolutional neural networks
( Section 7 ) retained state-of-the-art status in computer
vision and models resembling Sepp Hochreiterâs original design for the
LSTM recurrent neural network ( Section 10.1 ), dominated most
applications in natural language processing. Arguably, to that point,
the rapid emergence of deep learning appeared to be primarily
attributable to shifts in the available computational resources (thanks
to innovations in parallel computing with GPUs) and the availability of
massive data resources (thanks to cheap storage and Internet services).
While these factors may indeed remain the primary drivers behind this
technologyâs increasing power we are also witnessing, at long last, a
sea change in the landscape of dominant architectures.

At the present moment, the dominant models for nearly all natural
language processing tasks are based on the Transformer architecture.
Given any new task in natural language processing, the default
first-pass approach is to grab a large Transformer-based pretrained
model, (e.g., BERT ( Devlin et al. , 2018 ) , ELECTRA ( Clark et al. , 2020 ) , RoBERTa ( Liu et al. , 2019 ) , or
Longformer ( Beltagy et al. , 2020 ) ) adapting the output layers
as necessary, and fine-tuning the model on the available data for the
downstream task. If you have been paying attention to the last few years
of breathless news coverage centered on OpenAIâs large language models,
then you have been tracking a conversation centered on the GPT-2 and
GPT-3 Transformer-based models ( Brown et al. , 2020 , Radford et al. , 2019 ) . Meanwhile, the
vision Transformer has emerged as a default model for diverse vision
tasks, including image recognition, object detection, semantic
segmentation, and superresolution ( Dosovitskiy et al. , 2021 , Liu et al. , 2021 ) . Transformers
also showed up as competitive methods for speech recognition ( Gulati et al. , 2020 ) , reinforcement learning ( Chen et al. , 2021 ) , and graph neural networks ( Dwivedi and Bresson, 2020 ) .

The core idea behind the Transformer model is the attention mechanism ,
an innovation that was originally envisioned as an enhancement for
encoderâdecoder RNNs applied to sequence-to-sequence applications, such
as machine translations ( Bahdanau et al. , 2014 ) . You might
recall that in the first sequence-to-sequence models for machine
translation ( Sutskever et al. , 2014 ) , the entire input was
compressed by the encoder into a single fixed-length vector to be fed
into the decoder. The intuition behind attention is that rather than
compressing the input, it might be better for the decoder to revisit the
input sequence at every step. Moreover, rather than always seeing the
same representation of the input, one might imagine that the decoder
should selectively focus on particular parts of the input sequence at
particular decoding steps. Bahdanauâs attention mechanism provided a
simple means by which the decoder could dynamically attend to
different parts of the input at each decoding step. The high-level idea
is that the encoder could produce a representation of length equal to
the original input sequence. Then, at decoding time, the decoder can
(via some control mechanism) receive as input a context vector
consisting of a weighted sum of the representations on the input at each
time step. Intuitively, the weights determine the extent to which each
stepâs context âfocusesâ on each input token, and the key is to make
this process for assigning the weights differentiable so that it can be
learned along with all of the other neural network parameters.

Initially, the idea was a remarkably successful enhancement to the
recurrent neural networks that already dominated machine translation
applications. The models performed better than the original
encoderâdecoder sequence-to-sequence architectures. Furthermore,
researchers noted that some nice qualitative insights sometimes emerged
from inspecting the pattern of attention weights. In translation tasks,
attention models often assigned high attention weights to cross-lingual
synonyms when generating the corresponding words in the target language.
For example, when translating the sentence âmy feet hurtâ to âjâai mal
au piedsâ, the neural network might assign high attention weights to the
representation of âfeetâ when generating the corresponding French word
âpiedsâ. These insights spurred claims that attention models confer
âinterpretabilityâ although what precisely the attention weights
meanâi.e., how, if at all, they should be interpreted remains a hazy
research topic.

However, attention mechanisms soon emerged as more significant concerns,
beyond their usefulness as an enhancement for encoderâdecoder recurrent
neural networks and their putative usefulness for picking out salient
inputs. Vaswani et al. ( 2017 ) proposed the
Transformer architecture for machine translation, dispensing with
recurrent connections altogether, and instead relying on cleverly
arranged attention mechanisms to capture all relationships among input
and output tokens. The architecture performed remarkably well, and by
2018 the Transformer began showing up in the majority of
state-of-the-art natural language processing systems. Moreover, at the
same time, the dominant practice in natural language processing became
to pretrain large-scale models on enormous generic background corpora to
optimize some self-supervised pretraining objective, and then to
fine-tune these models using the available downstream data. The gap
between Transformers and traditional architectures grew especially wide
when applied in this pretraining paradigm, and thus the ascendance of
Transformers coincided with the ascendence of such large-scale
pretrained models, now sometimes called foundation models ( Bommasani et al. , 2021 ) .

In this chapter, we introduce attention models, starting with the most
basic intuitions and the simplest instantiations of the idea. We then
work our way up to the Transformer architecture, the vision Transformer,
and the landscape of modern Transformer-based pretrained models.
