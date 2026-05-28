---
topic: self-attention
generated_by: opus
generated_at: 2026-05-27T00:00:00Z
sources_used:
  - corpus/papers/1706.03762.md
  - corpus/web/jalammargithubio/jay-alammar-the-illustrated-transformer.md
  - corpus/web/lilianwenggithubio/lilian-weng-attention-attention.md
  - corpus/web/lilianwenggithubio/lilian-weng-the-transformer-family-v2.md
  - corpus/youtube/andrej-karpathy-neural-networks-zero-to-/kCc8FmEb1nY.md
  - corpus/papers/1810.04805.md
  - corpus/papers/1911.02150.md
related_topics: [multi-head-attention, positional-encoding, flash-attention, rope, masked-attention, transformer-block]
---

> **Example artifact.** This is a synthesized, fully-cited study note assembled from
> several documents in this corpus. It demonstrates use case #4 in the README:
> compressing multiple sources on one topic into a single note where every claim is
> traced back to its origin. Generated with an LLM from the listed `sources_used`.

# Self-Attention

A verbatim-extract topic doc. Quotes are reproduced unchanged from their sources, with only minor line-break reflowing where the source PDF parser inserted them mid-equation. Every quote is followed by a citation `(filename.md, line range or char offset)`.

---

## 1. Verbatim definitions (multiple framings)

> "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key."
— (1706.03762.md, lines 162-173)

> "Self-attention , also known as intra-attention , is an attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence. It has been shown to be very useful in machine reading, abstractive summarization, or image description generation."
— (lilian-weng-attention-attention.md, lines 164-166)

> "Self-attention is a type of attention mechanism where the model makes prediction for one part of a data sample using other parts of the observation about the same sample. Conceptually, it feels quite similar to non-local means . Also note that self-attention is permutation-invariant; in other words, it is an operation on sets."
— (lilian-weng-the-transformer-family-v2.md, line 80)

> "Self-Attention(&) — Relating different positions of the same input sequence. Theoretically the self-attention can adopt any score functions above, but just replace the target sequence with the same input sequence."
— (lilian-weng-attention-attention.md, lines 150-152). The footnote `(&)` reads: "Also, referred to as 'intra-attention' in Cheng et al., 2016 and some other papers." (line 162)

> "The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder."
— (1706.03762.md, lines 242-245)

> "As the model processes each word (each position in the input sequence), self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word. … Self-attention is the method the Transformer uses to bake the 'understanding' of other relevant words into the one we’re currently processing."
— (jay-alammar-the-illustrated-transformer.md, lines 78-80)

> "attention is a communication mechanism you can really think about it as a communication mechanism where you have a number of nodes in a directed graph where basically you have edges pointed between noes like this and what happens is every node has some Vector of information and it gets to aggregate information via a weighted sum from all of the nodes that point to it and this is done in a data dependent manner so depending on whatever data is actually stored that you should not at any point in time"
— (kCc8FmEb1nY.md, char ~66100, ≈1:10:40)

> "the way self attention solves this is the following every single node or every single token at each position will emit two vectors it will emit a query and it will emit a key now the query Vector roughly speaking is what am I looking for and the key Vector roughly speaking is what do I contain and then the way we get affinities between these uh tokens now in a sequence is we basically just do a do product between the keys and the queries so my query dot products with all the keys of all the other tokens and that dot product now becomes wayy and so um if the key and the query are sort of aligned they will interact to a very high amount and then I will get to learn more about that specific token as opposed to any other token in the sequence"
— (kCc8FmEb1nY.md, char ~59000, ≈1:03:00)

---

## 2. The canonical math (verbatim)

### Equation (1): Scaled Dot-Product Attention

> "We call our particular attention 'Scaled Dot-Product Attention' (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.
> In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:
>
> Attention(Q, K, V ) = softmax(QK^T / √dk) V    (1)"
— (1706.03762.md, lines 174-186)

### The two attention families and the scaling rationale (footnote 4)

> "The two most commonly used attention functions are additive attention [2], and dot-product (multi-plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√dk. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
> While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1/√dk."
— (1706.03762.md, lines 187-200)

> "4 To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Σ_{i=1..dk} q_i k_i, has mean 0 and variance dk."
— (1706.03762.md, lines 207-209)

### Element-wise (per-pair) score form

> "And for a query and a key vector $\mathbf{q}_i, \mathbf{k}_j \in \mathbb{R}^d$ (row vectors in query and key matrices), we have a scalar score:
>
> $$ a_{ij} = \text{softmax}\!\left(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}}\right) = \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top / \sqrt{d_k})}{ \sum_{r \in \mathcal{S}_i} \exp(\mathbf{q}_i {\mathbf{k}_r}^\top / \sqrt{d_k}) } $$
>
> where $\mathcal{S}_i$ is a collection of key positions for the $i$-th query to attend to."
— (lilian-weng-the-transformer-family-v2.md, lines 88-95)

### Lilian Weng's compact restatement

> "The transformer adopts the scaled dot-product attention : the output is a weighted sum of the values, where the weight assigned to each value is determined by the dot-product of the query with all the keys:
>
> $$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{n}}\right)\mathbf{V} $$"
— (lilian-weng-attention-attention.md, lines 301-305)

(Quoted because Lilian writes √n where Vaswani writes √d_k; the two are the same quantity, but this is a notational fork that level-designers should be aware of.)

### Multi-head context (kept brief; full deep-dive lives in `[[topic-multi-head-attention]]`)

> "MultiHead(Q, K, V ) = Concat(head1, ..., headh) W^O
> where head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
> Where the projections are parameter matrices W^Q_i ∈ R^{dmodel×dk}, W^K_i ∈ R^{dmodel×dk}, W^V_i ∈ R^{dmodel×dv} and W^O ∈ R^{hdv×dmodel}.
> In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality."
— (1706.03762.md, lines 218-233)

### Attention score taxonomy (for context)

Lilian Weng's table of attention score functions, verbatim row-by-row:

| Name | Alignment score function | Citation |
|---|---|---|
| Content-base attention | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \text{cosine}[\boldsymbol{s}_t, \boldsymbol{h}_i]$ | Graves2014 |
| Additive(\*) | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \mathbf{v}_a^\top \tanh(\mathbf{W}_a[\boldsymbol{s}_{t-1}; \boldsymbol{h}_i])$ | Bahdanau2015 |
| Location-Base | $\alpha_{t,i} = \text{softmax}(\mathbf{W}_a \boldsymbol{s}_t)$ | Luong2015 |
| General | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top \mathbf{W}_a \boldsymbol{h}_i$ | Luong2015 |
| Dot-Product | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top \boldsymbol{h}_i$ | Luong2015 |
| Scaled Dot-Product(^) | $\text{score}(\boldsymbol{s}_t, \boldsymbol{h}_i) = \boldsymbol{s}_t^\top \boldsymbol{h}_i / \sqrt{n}$ | Vaswani2017 |

> "(*) Referred to as 'concat' in Luong, et al., 2015 and as 'additive attention' in Vaswani, et al., 2017. (^) It adds a scaling factor $1/\sqrt{n}$, motivated by the concern when the input is large, the softmax function may have an extremely small gradient, hard for efficient learning."
— (lilian-weng-attention-attention.md, lines 107-142)

---

## 3. Step-by-step mechanism (extracted explanations)

### 3.1 Computing queries, keys, values

> "The first step in calculating self-attention is to create three vectors from each of the encoder’s input vectors (in this case, the embedding of each word). So for each word, we create a Query vector, a Key vector, and a Value vector. These vectors are created by multiplying the embedding by three matrices that we trained during the training process.
> Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant."
— (jay-alammar-the-illustrated-transformer.md, lines 91-94)

> "Multiplying x1 by the WQ weight matrix produces q1, the 'query' vector associated with that word. We end up creating a 'query', a 'key', and a 'value' projection of each word in the input sentence.
> What are the 'query', 'key', and 'value' vectors? They’re abstractions that are useful for calculating and thinking about attention. Once you proceed with reading how attention is calculated below, you’ll know pretty much all you need to know about the role each of these vectors plays."
— (jay-alammar-the-illustrated-transformer.md, lines 98-109)

> "The major component in the transformer is the unit of multi-head self-attention mechanism. The transformer views the encoded representation of the input as a set of key - value pairs, $(\mathbf{K}, \mathbf{V})$, both of dimension $n$ (input sequence length); in the context of NMT, both the keys and values are the encoder hidden states. In the decoder, the previous output is compressed into a query ($\mathbf{Q}$ of dimension $m$) and the next output is produced by mapping this query and the set of keys and values."
— (lilian-weng-attention-attention.md, lines 297-299)

> "so this attention here happens to be self attention but in principle um attention is a lot more General … the reason this attention is self attention is because because the keys queries and the values are all coming from the same Source from X so the same Source X produces Keys queries and values so these nodes are self attending"
— (kCc8FmEb1nY.md, char ~70000-70500, ≈1:14:30)

### 3.2 The dot-product score

> "The second step in calculating self-attention is to calculate a score. Say we’re calculating the self-attention for the first word in this example, 'Thinking'. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.
> The score is calculated by taking the dot product of the query vector with the key vector of the respective word we’re scoring. So if we’re processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1. The second score would be the dot product of q1 and k2."
— (jay-alammar-the-illustrated-transformer.md, lines 111-113)

> "all the queries will do product with all the keys so basically what we want is we want way now or the affinities between these to be query multiplying key but we have to be careful with uh we can't Matrix multiply this we actually need to transpose uh K but we have to be also careful because these are when you have The Bash Dimension so in particular we want to transpose uh the last two dimensions dimension1 and dimension -2 so -21 and so this Matrix multiply now will basically do the following B by T by 16 Matrix multiplies B by 16 by T to give us B by T by T right so for every row of B we're now going to have a t Square Matrix giving us the affinities"
— (kCc8FmEb1nY.md, char ~60500, ≈1:04:30)

### 3.3 The softmax + scaling

> "The third and fourth steps are to divide the scores by 8 (the square root of the dimension of the key vectors used in the paper – 64. This leads to having more stable gradients. There could be other possible values here, but this is the default), then pass the result through a softmax operation. Softmax normalizes the scores so they’re all positive and add up to 1.
> This softmax score determines how much each word will be expressed at this position. Clearly the word at this position will have the highest softmax score, but sometimes it’s useful to attend to another word that is relevant to the current word."
— (jay-alammar-the-illustrated-transformer.md, lines 115-117)

### 3.4 The weighted sum

> "The fifth step is to multiply each value vector by the softmax score (in preparation to sum them up). The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001, for example).
> The sixth step is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position (for the first word).
> That concludes the self-attention calculation. The resulting vector is one we can send along to the feed-forward neural network."
— (jay-alammar-the-illustrated-transformer.md, lines 119-123)

> "V is the elements that we aggregate or the the vectors that we aggregate instead of the raw X … you can think of X as kind of like private information to this token if you if you think about it that way so X is kind of private to this token so I'm a fifth token at some and I have some identity and uh my information is kept in Vector X and now for the purposes of the single head here's what I'm interested in here's what I have and if you find me interesting here's what I will communicate to you and that's stored in v and so V is the thing that gets aggregated for the purposes of this single head between the different notes and that's uh basically the self attention mechanism this is this is what it does"
— (kCc8FmEb1nY.md, char ~65400, ≈1:09:50)

### 3.5 The matrix form (condensed six steps)

> "Finally, since we’re dealing with matrices, we can condense steps two through six in one formula to calculate the outputs of the self-attention layer."
— (jay-alammar-the-illustrated-transformer.md, lines 136-138)

---

## 4. Code: verbatim implementations

### 4.1 Karpathy's "mathematical trick" — averaging past tokens via lower-triangular matmul (precursor to attention)

> "in other words, we're going to create X and B is short for bag of words because bag of words is um is kind of like um a term that people use when you are just averaging up things so this is just a bag of words basically there's a word stored on every one of these eight locations and we're doing a bag of words we're just averaging so in the beginning we're going to say that it's just initialized at Zero and then I'm doing a for Loop here so we're not being efficient yet that's coming but for now we're just iterating over all the batch Dimensions independently iterating over time and then the previous uh tokens are at this uh batch Dimension and then everything up to and including the teeth token"
— (kCc8FmEb1nY.md, char ~43100, ≈0:46:00)

The trick that follows: rewrite the average as a matrix multiply by a row-normalized lower-triangular matrix. Karpathy walks through three equivalent versions (for-loop, matmul with row-normalized lower-tri, and softmax(masked -inf)) and shows they all produce the same `xbow` tensor.

> "the third version and it's also identical to the first and second but let me talk through it it uses softmax so Trill here is this Matrix lower triangular ones way begins as all zero okay so if I just print way in the beginning it's all zero then I used masked fill so what this is doing is we. masked fill it's all zeros and I'm saying for all the elements where Trill is equal equal Z make them be negative Infinity so all the elements where Trill is zero will become negative Infinity now so this is what we get and then the final line here is softmax so if I take a softmax along every single so dim is negative one so along every single row if I do softmax what is that going to do well softmax is um is also like a normalization operation right and so spoiler alert you get the exact same Matrix"
— (kCc8FmEb1nY.md, char ~50800, ≈0:54:15)

> "the reason that this is a bit more interesting and the reason we're going to end up using it in self attention is that these weights here begin uh with zero and you can think of this as like an interaction strength or like an affinity so basically it's telling us how much of each uh token from the past do we want to Aggregate and average up and then this line is saying tokens from the past cannot communicate by setting them to negative Infinity we're saying that we will not aggregate anything from those tokens and so basically this then goes through softmax and through the weighted and this is the aggregation through matrix multiplication and so what this is now is you can think of these as um these zeros are currently just set by us to be zero but a quick preview is that these affinities between the tokens are not going to be just constant at zero they're going to be data dependent these tokens are going to start looking at each other and some tokens will find other tokens more or less interesting and depending on what their values are they're going to find each other interesting to different amounts"
— (kCc8FmEb1nY.md, char ~52200, ≈0:55:40)

### 4.2 Karpathy's single-head self-attention block (verbatim narration of the code)

> "I created this head module and it implements a single head of self attention so you give it a head size and then here it creates the key query and the value linear layers typically people don't use biases in these uh so those are the linear projections that we're going to apply to all of our nodes now here I'm creating this Trill variable Trill is not a parameter of the module so in sort of pytorch naming conventions uh this is called a buffer it's not a parameter and you have to call it you have to assign it to the module using a register buffer so that creates the trill uh the triang lower triangular Matrix and we're given the input X this should look very familiar now we calculate the keys the queries we C calculate the attention scores inside way uh we normalize it so we're using scaled attention here then we make sure that uh future doesn't communicate with the past so this makes it a decoder block and then softmax and then aggregate the value and output"
— (kCc8FmEb1nY.md, char ~73500, ≈1:18:30)

The full code Karpathy types (reconstructed verbatim from the narration he gives — Karpathy's nanoGPT `Head` class, single-head decoder self-attention):

```python
# from kCc8FmEb1nY.md (Karpathy "Let's build GPT", char ~58700-75000, ≈1:02:40-1:20:10)
# nanoGPT-style single head of masked self-attention

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is a buffer (not a parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)   note: scaled by 1/sqrt(head_size)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # decoder mask
        wei = F.softmax(wei, dim=-1)              # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)                          # (B, T, head_size)
        out = wei @ v                              # (B, T, head_size)
        return out
```

### 4.3 Shazeer's einsum reference — DotProductAttention on one query

```python
# from 1911.02150.md, lines 65-92 (Shazeer "Fast Transformer Decoding", §2.1)
def DotProductAttention(q, K, V):
    """Dot-Product Attention on one query.
    Args:
      q: a vector with shape [k]
      K: a matrix with shape [m, k]
      V: a matrix with shape [m, v]
    Returns:
      y: a vector with shape [v]
    """
    logits = tf.einsum("k,mk->m", q, K)
    weights = tf.softmax(logits)
    return tf.einsum("m,mv->v", weights, V)
```

> "Our code samples use einsum notation, as defined in TensorFlow and numpy, for generalized contractions between tensors of arbitrary dimension. In this notation, an equation names the dimensions of the input and output Tensors. The computation is numerically equivalent to broadcasting each input to have the union of all dimensions, multiplying component-wise, and summing across all dimensions not in the desired output shape."
— (1911.02150.md, lines 93-97)

### 4.4 Shazeer's einsum reference — MultiheadAttention on one query (vanilla MHA built on top of self-attention math)

```python
# from 1911.02150.md, lines 108-156 (Shazeer "Fast Transformer Decoding", §2.2)
def MultiheadAttention(x, M, P_q, P_k, P_v, P_o):
    """Multi-head Attention on one query.
    Args:
      x:   a vector with shape [d]
      M:   a matrix with shape [m, d]
      P_q: a tensor with shape [h, d, k]
      P_k: a tensor with shape [h, d, k]
      P_v: a tensor with shape [h, d, v]
      P_o: a tensor with shape [h, d, v]
    Returns:
      y: a vector with shape [d]
    """
    q = tf.einsum("d,hdk->hk", x, P_q)
    K = tf.einsum("md,hdk->hmk", M, P_k)
    V = tf.einsum("md,hdv->hmv", M, P_v)
    logits  = tf.einsum("hk,hmk->hm", q, K)
    weights = tf.softmax(logits)
    o = tf.einsum("hm,hmv->hv", weights, V)
    y = tf.einsum("hv,hdv->d", o, P_o)
    return y
# Note: [Vaswani et al., 2017] include a constant scaling factor on the logits.
# We omit this in our code, as it can be folded into the linear projections Pq or Pk.
```

(In the original transformer the input set `M` and the query source `x` are the same sequence — that's what makes it *self*-attention. Cross-attention is the case where they differ; see `[[topic-multi-head-attention]]`.)

---

## 5. Intuitions and analogies (verbatim from teachers)

### "It pronoun" — the canonical pedagogic example (Jay Alammar)

> "Say the following sentence is an input sentence we want to translate:
> ” The animal didn't cross the street because it was too tired ”
> What does 'it' in this sentence refer to? Is it referring to the street or to the animal? It’s a simple question to a human, but not as simple to an algorithm.
> When the model is processing the word 'it', self-attention allows it to associate 'it' with 'animal'."
— (jay-alammar-the-illustrated-transformer.md, lines 70-76)

> "If you’re familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it’s processing. Self-attention is the method the Transformer uses to bake the 'understanding' of other relevant words into the one we’re currently processing."
— (jay-alammar-the-illustrated-transformer.md, line 80)

### Attention as importance-weights (Lilian Weng)

> "attention in deep learning can be broadly interpreted as a vector of importance weights: in order to predict or infer one element, such as a pixel in an image or a word in a sentence, we estimate using the attention vector how strongly it is correlated with (or 'attends to' as you may have read in many papers) other elements and take the sum of their values weighted by the attention vector as the approximation of the target."
— (lilian-weng-attention-attention.md, line 25)

### Visual-attention analogy (Lilian Weng)

> "Human visual attention allows us to focus on a certain region with 'high resolution' (i.e. look at the pointy ear in the yellow box) while perceiving the surrounding image in 'low resolution' (i.e. now how about the snowy background and the outfit?), and then adjust the focal point or do the inference accordingly. Given a small patch of an image, pixels in the rest provide clues what should be displayed there. We expect to see a pointy ear in the yellow box because we have seen a dog’s nose, another pointy ear on the right, and Shiba’s mystery eyes (stuff in the red boxes). However, the sweater and blanket at the bottom would not be as helpful as those doggy features."
— (lilian-weng-attention-attention.md, line 18)

### Karpathy: communication channel + directed-graph view

> "attention is a communication mechanism you can really think about it as a communication mechanism where you have a number of nodes in a directed graph where basically you have edges pointed between noes like this and what happens is every node has some Vector of information and it gets to aggregate information via a weighted sum from all of the nodes that point to it and this is done in a data dependent manner"
— (kCc8FmEb1nY.md, char ~66100, ≈1:10:40)

> "in principle attention can be applied to any arbitrary directed graph and it's just a communication mechanism between the nodes"
— (kCc8FmEb1nY.md, char ~66800, ≈1:11:15)

> "the self attention is the communication and then once they've gathered all the data now they need to think on that data individually and so that's what feed forward is doing"
— (kCc8FmEb1nY.md, char ~80130, ≈1:25:30)
(Quoted because this "communication vs computation" framing is the signature Karpathy intuition for the transformer block.)

### Karpathy: query/key as "what am I looking for / what do I contain"

> "every single node or every single token at each position will emit two vectors it will emit a query and it will emit a key now the query Vector roughly speaking is what am I looking for and the key Vector roughly speaking is what do I contain"
— (kCc8FmEb1nY.md, char ~59100, ≈1:03:05)

### Karpathy: value as "what I will communicate to you"

> "you can think of X as kind of like private information to this token … so I'm a fifth token at some and I have some identity and uh my information is kept in Vector X and now for the purposes of the single head here's what I'm interested in here's what I have and if you find me interesting here's what I will communicate to you and that's stored in v"
— (kCc8FmEb1nY.md, char ~65500, ≈1:09:55)

### Karpathy: vowel-looking-for-consonant toy intuition

> "if I'm a vowel then maybe I'm looking for consonants in my past and maybe I want to know what those consonants are and I want that information to flow to me and so I want to now gather information from the past but I want to do it in the data dependent way and this is the problem that self attention solves"
— (kCc8FmEb1nY.md, char ~58400, ≈1:02:25)

### Karpathy: data-dependent affinities

> "these affinities between the tokens are not going to be just constant at zero they're going to be data dependent these tokens are going to start looking at each other and some tokens will find other tokens more or less interesting and depending on what their values are they're going to find each other interesting to different amounts and I'm going to call those affinities"
— (kCc8FmEb1nY.md, char ~52900, ≈0:56:20)

---

## 6. Why scaled dot-product (the √d_k)

### Vaswani's original argument (verbatim)

> "While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1/√dk."
— (1706.03762.md, lines 195-200)

> "4 To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Σ_{i=1..dk} q_i k_i, has mean 0 and variance dk."
— (1706.03762.md, lines 207-209)

### Karpathy's numerical demonstration (verbatim)

> "we've already implemented attention so given query key and value we've U multiplied the query and a key we've soft maxed it and then we are aggregating the values there's one more thing that we're missing here which is the dividing by one / square root of the head size the DK here is the head size why are they doing this finds this important so they call it the scaled attention and it's kind of like an important normalization to basically have the problem is if you have unit gsh and inputs so zero mean unit variance K and Q are unit gashin then if you just do we naively then you see that your we actually will be uh the variance will be on the order of head size which in our case is 16 but if you multiply by one over head size square root so this is square root and this is one over then the variance of we will be one so it will be preserved"
— (kCc8FmEb1nY.md, char ~71500, ≈1:16:30)

> "now why is this important you'll not notice that way here will feed into softmax and so it's really important especially at initialization that we be fairly diffuse … the problem is that because of softmax if weight takes on very positive and very negative numbers inside it softmax will actually converge towards one hot vectors and so I can illustrate that here um say we are applying softmax to a tensor of values that are very close to zero then we're going to get a diffuse thing out of softmax but the moment I take the exact same thing and I start sharpening it making it bigger by multiplying these numbers by eight for example you'll see that the softmax will start to sharpen and in fact it will sharpen towards the max so it will sharpen towards whatever number here is the highest and so um basically we don't want these values to be too extreme especially at initialization otherwise softmax will be way too peaky and um you're basically aggregating um information from like a single node every node just agregates information from a single other node that's not what we want especially at initialization and so the scaling is used just to control the variance at initialization"
— (kCc8FmEb1nY.md, char ~72100, ≈1:17:10)

### Lilian Weng's restatement (verbatim)

> "(^) It adds a scaling factor $1/\sqrt{n}$, motivated by the concern when the input is large, the softmax function may have an extremely small gradient, hard for efficient learning."
— (lilian-weng-attention-attention.md, line 142)

---

## 7. Why self-attention vs. recurrence (rationale)

Vaswani et al.'s "Why Self-Attention" section motivates the use of self-attention layers over recurrent or convolutional layers with three desiderata.

> "In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈ R^d, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.
> One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
> The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types."
— (1706.03762.md, lines 318-332)

### Table 1 (verbatim)

> "Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.
>
> | Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
> |---|---|---|---|
> | Self-Attention | O(n²·d) | O(1) | O(1) |
> | Recurrent | O(n·d²) | O(n) | O(n) |
> | Convolutional | O(k·n·d²) | O(1) | O(log_k(n)) |
> | Self-Attention (restricted) | O(r·n·d) | O(1) | O(n/r) |"
— (1706.03762.md, lines 273-296)

> "As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations."
— (1706.03762.md, lines 333-342)

> "To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work."
— (1706.03762.md, lines 342-345)

> "As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences."
— (1706.03762.md, lines 354-357)

---

## 8. Worked examples / illustrations

### 8.1 Anaphora resolution — Vaswani Figures 3, 4, 5 (verbatim figure captions)

> "Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'. Attentions here shown only for the word 'making'. Different colors represent different heads. Best viewed in color."
— (1706.03762.md, lines 878-881)

(The sentence visualized is: "It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult.")
— (1706.03762.md, lines 812-839)

> "Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word 'its' for attention heads 5 and 6. Note that the attentions are very sharp for this word."
— (1706.03762.md, lines 994-996)

(The sentence visualized for the anaphora is: "The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.")
— (1706.03762.md, lines 886-911)

> "Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks."
— (1706.03762.md, lines 1109-1111)

### 8.2 Jay Alammar's "it" example (verbatim)

> "As we are encoding the word 'it' in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on 'The Animal', and baked a part of its representation into the encoding of 'it'."
— (jay-alammar-the-illustrated-transformer.md, line 84)

> "As we encode the word 'it', one attention head is focusing most on 'the animal', while another is focusing on 'tired' -- in a sense, the model's representation of the word 'it' bakes in some of the representation of both 'animal' and 'tired'."
— (jay-alammar-the-illustrated-transformer.md, line 166)

### 8.3 Karpathy's lower-triangular toy worked example

> "I have a simple Matrix here that is a 3X3 of all ones a matrix B of just random numbers and it's a 3x2 and a matrix C which will be 3x3 multip 3x2 which will give out a 3x2 so here we're just using um matrix multiplication so a multiply B gives us C okay so how are these numbers in C um achieved right so this number in the top left is the first row of a dot product with the First Column of B and since all the the row of a right now is all just ones then the do product here with with this column of B is just going to do a sum of these of this column so 2 + 6 + 6 is 14 the element here in the output of C is also the first column here the first row of a multiplied now with the second column of B so 7 + 4 + 5 is 16 … the trick here uh the following this is just a boring number of um it's just a boring array of all ones but torch has this function called Trail … and you can wrap it in torch up once and it will just return the lower triangular portion of this … and because this is one and then zeros we what ended up happening is we're just plucking out the row of this row of B and that's what we got now here we have one 1 Z so here 110 do product with these two columns will now give us 2 + 6 which is 8 and 7 + 4 which is 11 and because this is 111 we ended up with the addition of all of them and so basically depending on how many ones and zeros we have here we are basically doing a sum currently of a variable number of these rows and that gets deposited into C"
— (kCc8FmEb1nY.md, char ~44400-46500, ≈0:47:25-0:49:30)

> "if we took a and then we did aals aide torch. sum in the um of a in the um oneth Dimension and then let's keep them as true so so therefore the broadcasting will work out so if I rerun this you see now that these rows now sum to one so this row is one this row is 0. 5.5 Z and here we get 1/3 and now when we do a multiply B what are we getting here we are just getting the first row first row here now we are getting the average of the first two rows okay so 2 and six average is four and four and seven average is 5.5 and on the bottom here we are now getting the average of these three rows so the average of all of elements of B are now deposited here"
— (kCc8FmEb1nY.md, char ~47200, ≈0:50:15)

### 8.4 Numerical demonstration of softmax sharpening (Karpathy)

> "say we are applying softmax to a tensor of values that are very close to zero then we're going to get a diffuse thing out of softmax but the moment I take the exact same thing and I start sharpening it making it bigger by multiplying these numbers by eight for example you'll see that the softmax will start to sharpen and in fact it will sharpen towards the max so it will sharpen towards whatever number here is the highest"
— (kCc8FmEb1nY.md, char ~72500, ≈1:17:35)

---

## 9. Common pitfalls / surprising remarks (verbatim)

### Self-attention is permutation-invariant — you MUST add positional information

> "Self-attention … is permutation-invariant; in other words, it is an operation on sets."
— (lilian-weng-the-transformer-family-v2.md, line 80)

> "notice that there is no notion of space so attention simply acts over like a set of vectors in this graph and so by default these nodes have no idea where they are positioned in the space and that's why we need to encode them positionally and sort of give them some information that is anchored to a specific position so that they sort of know where they are and this is different than for example from convolution because if you're run for example a convolution operation over some input there's a very specific sort of layout of the information in space and the convolutional filters sort of act in space and so it's it's not like an attention in ATT ention is just a set of vectors out there in space they communicate and if you want them to have a notion of space you need to specifically add it which is what we've done when we calculated the um relative the positional encode encodings and added that information to the vectors"
— (kCc8FmEb1nY.md, char ~67400, ≈1:11:55)
(See `[[topic-positional-encoding]]`.)

### Across batch elements, nothing communicates

> "the elements across the batch Dimension which are independent examples never talk to each other they're always processed independently and this is a batched matrix multiply that applies basically a matrix multiplication uh kind of in parallel across the batch dimension so maybe it would be more accurate to say that in this analogy of a directed graph we really have because the back size is four we really have four separate pools of eight nodes and those eight nodes only talk to each other"
— (kCc8FmEb1nY.md, char ~68500, ≈1:12:55)

### Encoder vs. decoder self-attention — the mask is the only difference

> "in the case of language modeling uh we have this specific uh structure of directed graph where the future tokens will not communicate to the Past tokens but this doesn't necessarily have to be the constraint in the general case and in fact in many cases you may want to have all of the uh noes talk to each other uh fully so as an example if you're doing sentiment analysis or something like that with a Transformer you might have a number of tokens and you may want to have them all talk to each other fully because later you are predicting for example the sentiment of the sentence and so it's okay for these NOS to talk to each other and so in those cases you will use an encoder block of self attention and uh all it means that it's an encoder block is that you will delete this line of code allowing all the noes to completely talk to each other what we're implementing here is sometimes called a decoder block and it's called a decoder because it is sort of like a decoding language and it's got this autor regressive format where you have to mask with the Triangular Matrix so that uh nodes from the future never talk to the Past because they would give away the answer"
— (kCc8FmEb1nY.md, char ~69200, ≈1:13:40)

### Self vs. cross attention — what makes it "self"

> "you keep hearing me say attention self attention Etc there's actually also something called cross attention what is the difference so basically the reason this attention is self attention is because because the keys queries and the values are all coming from the same Source from X so the same Source X produces Keys queries and values so these nodes are self attending but in principle attention is much more General than that so for example an encoder decoder Transformers uh you can have a case where the queries are produced from X but the keys and the values come from a whole separate external source"
— (kCc8FmEb1nY.md, char ~69900, ≈1:14:25)

### The mask is implemented inside the softmax

> "we modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i."
— (1706.03762.md, lines 156-159)

> "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections."
— (1706.03762.md, lines 246-250)

> "In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf ) before the softmax step in the self-attention calculation."
— (jay-alammar-the-illustrated-transformer.md, line 220)

### BERT distinction — bidirectional vs. constrained (autoregressive) self-attention

> "BERT_BASE was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left."
— (1810.04805.md, lines 360-365)

### Multi-head necessity — without it, "averaging inhibits" focus

> "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this."
— (1706.03762.md, lines 216-217)

> "It expands the model’s ability to focus on different positions. Yes, in the example above, z1 contains a little bit of every other encoding, but it could be dominated by the actual word itself. If we’re translating a sentence like 'The animal didn’t cross the street because it was too tired', it would be useful to know which word 'it' refers to."
— (jay-alammar-the-illustrated-transformer.md, line 146)

### Q, K, V dimensions don't *have* to be smaller than d_model (architecture choice for cheap multi-head)

> "Notice that these new vectors are smaller in dimension than the embedding vector. Their dimensionality is 64, while the embedding and encoder input/output vectors have dimensionality of 512. They don’t HAVE to be smaller, this is an architecture choice to make the computation of multiheaded attention (mostly) constant."
— (jay-alammar-the-illustrated-transformer.md, line 94)

### Karpathy: self-attention can't tolerate very high learning rates

> "I also came up to the script here and I decreased the learning rate because uh the self attention can't tolerate very very high learning rates"
— (kCc8FmEb1nY.md, char ~75300, ≈1:20:05)

---

## 10. Cross-references to related topic docs

- Multi-head attention: `[[topic-multi-head-attention]]` — self-attention is run h times in parallel with different learned projections, then concatenated and projected by W^O. The original paper uses h=8, dk=dv=dmodel/h=64. (See §2 of this doc for the verbatim MultiHead formula.)
- Masked attention: `[[topic-masked-attention]]` — the decoder variant. Implemented inside scaled dot-product attention by setting illegal logit positions to −∞ before the softmax. (See §9.)
- Positional encoding: `[[topic-positional-encoding]]` — required because self-attention is permutation-invariant. The original transformer uses sinusoidal PE; later work uses learned, RoPE, ALiBi, etc.
- RoPE: `[[topic-rope]]` — rotary position embedding that injects relative position into Q and K by rotation.
- Cross-attention / encoder-decoder attention: covered briefly in §3.1 and §9 here. Same math, but Q comes from one source and K, V from another.
- Efficient attention variants: `[[topic-flash-attention]]`, `[[topic-grouped-query-attention]]`, `[[topic-mla]]` — all leave the math here unchanged and optimize memory bandwidth or KV-cache.
- The transformer block: `[[topic-transformer-block]]` — wraps self-attention with residual + LayerNorm + position-wise FFN. Karpathy's "communication then computation" framing (see §5).
- Attention precursors: `[[topic-bahdanau-attention]]` — additive attention over RNN hidden states, the parent mechanism.

---

## 11. Source coverage

| Source file | Sections it contributed to | Notes |
|---|---|---|
| 1706.03762.md (Vaswani et al.) | §1, §2 (Eq. 1, MHA), §6 (√d_k footnote 4), §7 (Why Self-Attention + Table 1), §8 (Figs 3-5 anaphora), §9 (masking, multi-head necessity) | Canonical math + canonical rationale. Sole source for the variance-grows-with-d_k footnote and Table 1. |
| jay-alammar-the-illustrated-transformer.md | §1 (definitions), §3 (six-step walkthrough), §5 ("it" example), §8 (decoder mask), §9 (Q/K/V dimensions note) | Pedagogic walkthrough. The "it" pronoun example is canonical here. |
| lilian-weng-attention-attention.md | §1 (intra-attention definition), §2 (score taxonomy table), §5 (importance-weights, Shiba Inu analogy) | Score-function taxonomy is unique to this source; bridge between Bahdanau and Vaswani. |
| lilian-weng-the-transformer-family-v2.md | §1 (permutation-invariance), §2 (per-pair scalar score formula) | Provides the explicit per-element a_{ij} formula that complements Vaswani's matrix form. |
| kCc8FmEb1nY.md (Karpathy) | §1 (communication framing), §3 (Q/K/V narrated), §4.1-4.2 (matmul trick + Head class), §5 (vowel/consonant intuition, communication-vs-computation), §6 (numerical √d_k demo), §8 (lower-tri toy + softmax sharpening), §9 (no-batch-comm, encoder vs decoder mask, self vs cross) | Sole source for the "lower-triangular matmul = averaging" derivation, the directed-graph framing, and the numerical demonstration of softmax saturation. |
| 1810.04805.md (BERT) | §9 (bidirectional vs. constrained self-attention) | Only used for the BERT-vs-GPT self-attention distinction. |
| 1911.02150.md (Shazeer MQA) | §4.3-4.4 (einsum reference code) | Vanilla DotProductAttention + MultiheadAttention reference TF code. |
