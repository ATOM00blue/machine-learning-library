---
title: "Understanding Convolutions on Graphs"
source: "web"
url: "https://distill.pub/2021/understanding-gnns/"
domain: "distill.pub"
name: "Distill - Understanding Convolutions on Graphs"
fetched_at: "2026-05-26T23:30:24Z"
topics: ["gnn"]
---

This article is one of two Distill publications about graph neural networks.
        Take a look at A Gentle Introduction to Graph Neural Networks for a companion view on many things graph and neural network related.

Many systems and interactions - social networks, molecules, organizations, citations, physical models, transactions - can be represented quite naturally as graphs.
        How can we reason about and make predictions within these systems?

One idea is to look at tools that have worked well in other domains: neural networks have shown immense predictive power in a variety of learning tasks.
        However, neural networks have been traditionally used to operate on fixed-size and/or regular-structured inputs (such as sentences, images and video).
        This makes them unable to elegantly process graph-structured data.

![Neural networks generally operate on fixed-size input vectors. How do we input a graph to a neural network?](images/standard-neural-networks.svg)

Graph neural networks (GNNs) are a family of neural networks that can operate naturally on graph-structured data. 
      By extracting and utilizing features from the underlying graph,
      GNNs can make more informed predictions about entities in these interactions,
      as compared to models that consider individual entities in isolation.

GNNs are not the only tools available to model graph-structured data:
      graph kernels and random-walk methods were some of the most popular ones.
      Today, however, GNNs have largely replaced these techniques
      because of their inherent flexibility to model the underlying systems
      better.

In this article, we will illustrate
      the challenges of computing over graphs, 
      describe the origin and design of graph neural networks,
      and explore the most popular GNN variants in recent times.
      Particularly, we will see that many of these variants
      are composed of similar building blocks.

First, let’s discuss some of the complications that graphs come with.

## The Challenges of Computation on Graphs

### Lack of Consistent Structure

Graphs are extremely flexible mathematical models; but this means they lack consistent structure across instances.
        Consider the task of predicting whether a given chemical molecule is toxic :

![The molecular structure of non-toxic 1,2,6-trigalloyl-glucose.](images/1,2,6-trigalloyl-glucose-molecule.svg)
![The molecular structure of toxic caramboxin.](images/caramboxin-molecule.svg)

Left: A non-toxic 1,2,6-trigalloyl-glucose molecule.

Right: A toxic caramboxin molecule.

Looking at a few examples, the following issues quickly become apparent:

- Molecules may have different numbers of atoms.

- The atoms in a molecule may be of different types.

- Each of these atoms may have different number of connections.

- These connections can have different strengths.

Representing graphs in a format that can be computed over is non-trivial,
        and the final representation chosen often depends significantly on the actual problem.

### Node-Order Equivariance

Extending the point above: graphs often have no inherent ordering present amongst the nodes.
      Compare this to images, where every pixel is uniquely determined by its absolute position within the image!

![Representing the graph as one vector requires us to fix an order on the nodes. But what do we do when the nodes have no inherent order?](images/node-order-alternatives.svg)

          Representing the graph as one vector requires us to fix an order on the nodes.
          But what do we do when the nodes have no inherent order?
          Above: 
          The same graph labelled in two different ways. The alphabets indicate the ordering of the nodes.
        

As a result, we would like our algorithms to be node-order equivariant:
      they should not depend on the ordering of the nodes of the graph.
      If we permute the nodes in some way, the resulting representations of 
      the nodes as computed by our algorithms should also be permuted in the same way.

### Scalability

Graphs can be really large! Think about social networks like Facebook and Twitter, which have over a billion users. 
        Operating on data this large is not easy.

Luckily, most naturally occuring graphs are ‘sparse’:
        they tend to have their number of edges linear in their number of vertices.
        We will see that this allows the use of clever methods
        to efficiently compute representations of nodes within the graph.
        Further, the methods that we look at here will have significantly fewer parameters
        in comparison to the size of the graphs they operate on.

## Problem Setting and Notation

There are many useful problems that can be formulated over graphs:

- Node Classification: Classifying individual nodes.

- Graph Classification: Classifying entire graphs.

- Node Clustering: Grouping together similar nodes based on connectivity.

- Link Prediction: Predicting missing links.

- Influence Maximization: Identifying influential nodes.

![Examples of problems that can be defined over graphs.](images/graph-tasks.svg)

          Examples of problems that can be defined over graphs.
          This list is not exhaustive!
        

A common precursor in solving many of these problems is node representation learning :
        learning to map individual nodes to fixed-size real-valued vectors (called ‘representations’ or ‘embeddings’).

In Learning GNN Parameters , we will see how the learnt embeddings can be used for these tasks.

Different GNN variants are distinguished by the way these representations are computed.
        Generally, however, GNNs compute node representations in an iterative process.
        We will use the notation h v ( k ) h_v^{(k)} h v ( k ) ​ to indicate the representation of node v v v after the k th k^{\text{th}} k th iteration.
        Each iteration can be thought of as the equivalent of a ‘layer’ in standard neural networks.

We will define a graph G G G as a set of nodes, V V V , with a set of edges E E E connecting them.
        Nodes can have individual features as part of the input: we will denote by x v x_v x v ​ the individual feature for node v ∈ V v \in V v ∈ V .
        For example, the ‘node features’ for a pixel in a color image
        would be the red, green and blue channel (RGB) values at that pixel.

For ease of exposition, we will assume G G G is undirected, and all nodes are of the same type. These kinds of graphs are called ‘homogeneous’. Many of the same ideas we will see here 
        apply to other kinds of graphs:
        we will discuss this later in Different Kinds of Graphs .

Sometimes we will need to denote a graph property by a matrix M M M ,
        where each row M v M_v M v ​ represents a property corresponding to a particular vertex v v v .

## Extending Convolutions to Graphs

Convolutional Neural Networks have been seen to be quite powerful in extracting features from images.
      However, images themselves can be seen as graphs with a very regular grid-like structure,
      where the individual pixels are nodes, and the RGB channel values at each pixel as the node features.

A natural idea, then, is to consider generalizing convolutions to arbitrary graphs. Recall, however, the challenges
      listed out in the previous section : in particular, ordinary convolutions are not node-order invariant, because
      they depend on the absolute positions of pixels.
      It is initially unclear as how to generalize convolutions over grids to convolutions over general graphs,
      where the neighbourhood structure differs from node to node. The curious reader may wonder if performing some sort of padding and ordering
        could be done to ensure the consistency of neighbourhood structure across nodes.
        This has been attempted with some success ,
        but the techniques we will look at here are more general and powerful.

          Convolutions in CNNs are inherently localized.
          Neighbours participating in the convolution at the center pixel are highlighted in gray.
        

          GNNs can perform localized convolutions mimicking CNNs.
          Hover over a node to see its immediate neighbourhood highlighted on the left.
          The structure of this neighbourhood changes from node to node.
        

We begin by introducing the idea of constructing polynomial filters over node neighbourhoods,
      much like how CNNs compute localized filters over neighbouring pixels.
      Then, we will see how more recent approaches extend on this idea with more powerful mechanisms.
      Finally, we will discuss alternative methods
      that can use ‘global’ graph-level information for computing node representations.

## Polynomial Filters on Graphs

### The Graph Laplacian

Given a graph G G G , let us fix an arbitrary ordering of the n n n nodes of G G G .
        We denote the 0 − 1 0-1 0 − 1 adjacency matrix of G G G by A A A , we can construct the diagonal degree matrix D D D of G G G as:

Dv=∑uAvu.
          D_v = \sum_u A_{vu}.
        Dv​=u∑​Avu​.

            The degree of node vvv is the number of edges incident at vvv.
          

where A v u A_{vu} A v u ​ denotes the entry in the row corresponding to v v v and the column corresponding to u u u in the matrix A A A . We will use this notation throughout this section.

Then, the graph Laplacian L L L is the square n × n n \times n n × n matrix defined as: L = D − A . L = D - A. L = D − A .

![](images/laplacian.svg)

          The Laplacian LLL for an undirected graph GGG, with the row corresponding to node C\textsf{C}C highlighted.
          Zeros in LLL are not displayed above.
          The Laplacian LLL depends only on the structure of the graph GGG, not on any node features.
        

The graph Laplacian gets its name from being the discrete analog of the Laplacian operator from calculus.

Although it encodes precisely the same information as the adjacency matrix A A A In the sense that given either of the matrices A A A or L L L , you can construct the other. ,
        the graph Laplacian has many interesting properties of its own. The graph Laplacian shows up in many mathematical problems involving graphs: random walks , spectral clustering ,
          and diffusion , to name a few. We will see some of these properties
        in a later section ,
        but will instead point readers to this tutorial for greater insight into the graph Laplacian.

### Polynomials of the Laplacian

Now that we have understood what the graph Laplacian is,
        we can build polynomials of the form: p w ( L ) = w 0 I n + w 1 L + w 2 L 2 + … + w d L d = ∑ i = 0 d w i L i . p_w(L) = w_0 I_n + w_1 L + w_2 L^2 + \ldots + w_d L^d = \sum_{i = 0}^d w_i L^i. p w ​ ( L ) = w 0 ​ I n ​ + w 1 ​ L + w 2 ​ L 2 + … + w d ​ L d = i = 0 ∑ d ​ w i ​ L i . Each polynomial of this form can alternately be represented by
        its vector of coefficients w = [ w 0 , … , w d ] w = [w_0, \ldots, w_d] w = [ w 0 ​ , … , w d ​ ] .
        Note that for every w w w , p w ( L ) p_w(L) p w ​ ( L ) is an n × n n \times n n × n matrix, just like L L L .

These polynomials can be thought of as the equivalent of ‘filters’ in CNNs,
        and the coefficients w w w as the weights of the ‘filters’.

For ease of exposition, we will focus on the case where nodes have one-dimensional features:
        each of the x v x_v x v ​ for v ∈ V v \in V v ∈ V is just a real number. 
        The same ideas hold when each of the x v x_v x v ​ are higher-dimensional vectors, as well.

Using the previously chosen ordering of the nodes,
        we can stack all of the node features x v x_v x v ​ to get a vector x ∈ R n x \in \mathbb{R}^n x ∈ R n .

![Fixing a node order and collecting all node features into a single vector.](images/node-order-vector.svg)

          Fixing a node order (indicated by the alphabets) and collecting all node features into a single vector xxx.
        

Once we have constructed the feature vector x x x ,
        we can define its convolution with a polynomial filter p w p_w p w ​ as: x ′ = p w ( L ) x x’ = p_w(L) \ x x ′ = p w ​ ( L ) x To understand how the coefficients w w w affect the convolution,
        let us begin by considering the ‘simplest’ polynomial:
        when w 0 = 1 w_0 = 1 w 0 ​ = 1 and all of the other coefficients are 0 0 0 .
        In this case, x ′ x’ x ′ is just x x x : x ′ = p w ( L ) x = ∑ i = 0 d w i L i x = w 0 I n x = x . x’ = p_w(L) \ x = \sum_{i = 0}^d w_i L^ix = w_0 I_n x = x. x ′ = p w ​ ( L ) x = i = 0 ∑ d ​ w i ​ L i x = w 0 ​ I n ​ x = x . Now, if we increase the degree, and consider the case where
        instead w 1 = 1 w_1 = 1 w 1 ​ = 1 and and all of the other coefficients are 0 0 0 .
        Then, x ′ x’ x ′ is just L x Lx L x , and so: x v ′ = ( L x ) v = L v x = ∑ u ∈ G L v u x u = ∑ u ∈ G ( D v u − A v u ) x u = D v x v − ∑ u ∈ N ( v ) x u \begin{aligned}
           x’_v = (Lx)_v &= L_v x \\ 
                         &= \sum_{u \in G} L_{vu} x_u \\ 
                         &= \sum_{u \in G} (D_{vu} - A_{vu}) x_u \\ 
                         &= D_v \ x_v - \sum_{u \in \mathcal{N}(v)} x_u
          \end{aligned} x v ′ ​ = ( L x ) v ​ ​ = L v ​ x = u ∈ G ∑ ​ L v u ​ x u ​ = u ∈ G ∑ ​ ( D v u ​ − A v u ​ ) x u ​ = D v ​ x v ​ − u ∈ N ( v ) ∑ ​ x u ​ ​ We see that the features at each node v v v are combined
        with the features of its immediate neighbours u ∈ N ( v ) u \in \mathcal{N}(v) u ∈ N ( v ) . For readers familiar with Laplacian filtering of images ,
          this is the exact same idea. When x x x is an image, x ′ = L x x’ = Lx x ′ = L x is exactly the result of applying a ‘Laplacian filter’ to x x x .

At this point, a natural question to ask is:
        How does the degree d d d of the polynomial influence the behaviour of the convolution?
        Indeed, it is not too hard to show that: This is Lemma 5.2 from . dist G ( v , u ) > i ⟹ L v u i = 0 . \text{dist}_G(v, u) > i \quad \Longrightarrow \quad L_{vu}^i = 0. dist G ​ ( v , u ) > i ⟹ L v u i ​ = 0 . This implies, when we convolve x x x with p w ( L ) p_w(L) p w ​ ( L ) of degree d d d to get x ′ x’ x ′ : x v ′ = ( p w ( L ) x ) v = ( p w ( L ) ) v x = ∑ i = 0 d w i L v i x = ∑ i = 0 d w i ∑ u ∈ G L v u i x u = ∑ i = 0 d w i ∑ u ∈ G dist G ( v , u ) ≤ i L v u i x u . \begin{aligned}
          x’_v = (p_w(L)x)_v  &= (p_w(L))_v x \\
              &= \sum_{i = 0}^d w_i L_v^i x \\
              &= \sum_{i = 0}^d w_i \sum_{u \in G} L_{vu}^i x_u \\
              &= \sum_{i = 0}^d w_i \sum_{u \in G \atop \text{dist}_G(v, u) \leq i} L_{vu}^i x_u.
          \end{aligned} x v ′ ​ = ( p w ​ ( L ) x ) v ​ ​ = ( p w ​ ( L ) ) v ​ x = i = 0 ∑ d ​ w i ​ L v i ​ x = i = 0 ∑ d ​ w i ​ u ∈ G ∑ ​ L v u i ​ x u ​ = i = 0 ∑ d ​ w i ​ dist G ​ ( v , u ) ≤ i u ∈ G ​ ∑ ​ L v u i ​ x u ​ . ​

Effectively, the convolution at node v v v occurs only with nodes u u u which are not more than d d d hops away.
        Thus, these polynomial filters are localized. The degree of the localization is governed completely by d d d .

To help you understand these ‘polynomial-based’ convolutions better, we have created the visualization below.
        Vary the polynomial coefficients and the input grid x x x to see how the result x ′ x’ x ′ of the convolution changes.
        The grid under the arrow shows the equivalent convolutional kernel applied at the highlighted pixel in x x x to get
        the resulting pixel in x ′ x’ x ′ .
        The kernel corresponds to the row of p w ( L ) p_w(L) p w ​ ( L ) for the highlighted pixel.
        Note that even after adjusting for position,
        this kernel is different for different pixels, depending on their position within the grid.

Hover over a pixel in the input grid (left, representing x x x )
              to highlight it and see the equivalent convolutional kernel
              for that pixel under the arrow.
              The result x ′ x’ x ′ of the convolution is shown on the right:
              note that different convolutional kernels are applied at different pixels,
              depending on their location.

Click on the input grid to toggle pixel values between 0 0 0 (white) and 1 1 1 (blue).
              To randomize the input grid, press ‘Randomize Grid’. To reset all pixels to 0 0 0 , press ‘Reset Grid’.
              Use the sliders at the bottom to change the coefficients w w w .
              To reset all coefficients w w w to 0 0 0 , press ‘Reset Coefficients.’

### ChebNet

        ChebNet  refines this idea of polynomial filters by looking at polynomial filters of the form:
      
pw(L)=∑i=1dwiTi(L~)
          p_w(L) = \sum_{i = 1}^d w_i T_i(\tilde{L})
        pw​(L)=i=1∑d​wi​Ti​(L~)

        where TiT_iTi​ is the degree-iii
Chebyshev polynomial of the first kind and
        L~\tilde{L}L~ is the normalized Laplacian defined using the largest eigenvalue of LLL:
        
          We discuss the eigenvalues of the Laplacian LLL in more detail in a later section.
        

L~=2Lλmax(L)−In.
          \tilde{L} = \frac{2L}{\lambda_{\max}(L)} - I_n.
        L~=λmax​(L)2L​−In​.

What is the motivation behind these choices?

- L L L is actually positive semi-definite: all of the eigenvalues of L L L are not lesser than 0 0 0 .
            If λ max ( L ) > 1 \lambda_{\max}(L) > 1 λ max ​ ( L ) > 1 , the entries in the powers of L L L rapidly increase in size. L ~ \tilde{L} L ~ is effectively a scaled-down version of L L L , with eigenvalues guaranteed to be in the range [ − 1 , 1 ] [-1, 1] [ − 1 , 1 ] .
            This prevents the entries of powers of L ~ \tilde{L} L ~ from blowing up.
            Indeed, in the visualization above : we restrict the higher-order coefficients
            when the unnormalized Laplacian L L L is selected, but allow larger values when the normalized Laplacian L ~ \tilde{L} L ~ is selected,
            in order to show the result x ′ x’ x ′ on the same color scale.

- The Chebyshev polynomials have certain interesting properties that make interpolation more numerically stable.
            We won’t talk about this in more depth here,
            but will advise interested readers to take a look at as a definitive resource.

### Polynomial Filters are Node-Order Equivariant

The polynomial filters we considered here are actually independent of the ordering of the nodes.
        This is particularly easy to see when the degree of the polynomial p w p_w p w ​ is 1 1 1 :
        where each node’s feature is aggregated with the sum of its neighbour’s features.
        Clearly, this sum does not depend on the order of the neighbours.
        A similar proof follows for higher degree polynomials:
        the entries in the powers of L L L are equivariant to the ordering of the nodes.

Details for the Interested Reader

As above, let’s assume an arbitrary node-order over the n n n nodes of our graph.
          Any other node-order can be thought of as a permutation of this original node-order.
          We can represent any permutation by a permutation matrix P P P . P P P will always be an orthogonal 0 − 1 0-1 0 − 1 matrix: P P T = P T P = I n . PP^T = P^TP = I_n. P P T = P T P = I n ​ . Then, we call a function f f f node-order equivariant iff for all permutations P P P : f ( P x ) = P f ( x ) . f(Px) = P f(x). f ( P x ) = P f ( x ) . When switching to the new node-order using the permutation P P P ,
          the quantities below transform in the following way: x → P x L → P L P T L i → P L i P T \begin{aligned}
              x &\to Px \\
              L &\to PLP^T \\
              L^i &\to PL^iP^T
            \end{aligned} x L L i ​ → P x → P L P T → P L i P T ​ and so, for the case of polynomial filters where f ( x ) = p w ( L ) x f(x) = p_w(L) \ x f ( x ) = p w ​ ( L ) x , we can see that: f ( P x ) = ∑ i = 0 d w i ( P L i P T ) ( P x ) = P ∑ i = 0 d w i L i x = P f ( x ) . \begin{aligned}
              f(Px) & = \sum_{i = 0}^d w_i (PL^iP^T) (Px) \\
                    & = P \sum_{i = 0}^d w_i L^i x \\
                    & = P f(x).
            \end{aligned} f ( P x ) ​ = i = 0 ∑ d ​ w i ​ ( P L i P T ) ( P x ) = P i = 0 ∑ d ​ w i ​ L i x = P f ( x ) . ​ as claimed.

### Embedding Computation

We now describe how we can build a graph neural network
          by stacking ChebNet (or any polynomial filter) layers
          one after the other with non-linearities,
          much like a standard CNN.
          In particular, if we have K K K different polynomial filter layers,
          the k th k^{\text{th}} k th of which has its own learnable weights w ( k ) w^{(k)} w ( k ) ,
          we would perform the following computation:

Note that these networks
          reuse the same filter weights across different nodes,
          exactly mimicking weight-sharing in Convolutional Neural Networks (CNNs)
          which reuse weights for convolutional filters across a grid.

## Modern Graph Neural Networks

ChebNet was a breakthrough in learning localized filters over graphs,
        and it motivated many to think of graph convolutions from a different perspective.

        We return back to the result of convolving xxx by the polynomial kernel pw(L)=Lp_w(L) = Lpw​(L)=L,
        focussing on a particular vertex vvv:
      
(Lx)v=Lvx=∑u∈GLvuxu=∑u∈G(Dvu−Avu)xu=Dv xv−∑u∈N(v)xu
          \begin{aligned}
           (Lx)_v &= L_v x \\ 
                  &= \sum_{u \in G} L_{vu} x_u \\ 
                  &= \sum_{u \in G} (D_{vu} - A_{vu}) x_u \\ 
                  &= D_v \  x_v - \sum_{u \in \mathcal{N}(v)} x_u
          \end{aligned}
        (Lx)v​​=Lv​x=u∈G∑​Lvu​xu​=u∈G∑​(Dvu​−Avu​)xu​=Dv​ xv​−u∈N(v)∑​xu​​

As we noted before, this is a 1 1 1 -hop localized convolution.
        But more importantly, we can think of this convolution as arising of two steps:

- Aggregating over immediate neighbour features x u x_u x u ​ .

- Combining with the node’s own feature x v x_v x v ​ .

Key Idea: What if we consider different kinds of ‘aggregation’ and ‘combination’ steps,
        beyond what are possible using polynomial filters?

By ensuring that the aggregation is node-order equivariant,
        the overall convolution becomes node-order equivariant.

These convolutions can be thought of as ‘message-passing’ between adjacent nodes:
        after each step, every node receives some ‘information’ from its neighbours.

By iteratively repeating the 1 1 1 -hop localized convolutions K K K times (i.e., repeatedly ‘passing messages’),
        the receptive field of the convolution effectively includes all nodes upto K K K hops away.

### Embedding Computation

Message-passing forms the backbone of many GNN architectures today.
          We describe the most popular ones in depth below:

- Graph Convolutional Networks (GCN)

- Graph Attention Networks (GAT)

- Graph Sample and Aggregate (GraphSAGE)

- Graph Isomorphism Network (GIN)

### Thoughts

An interesting point is to assess different aggregation functions: are some better and others worse? demonstrates that aggregation functions indeed can be compared on how well
        they can uniquely preserve node neighbourhood features;
        we recommend the interested reader take a look at the detailed theoretical analysis there.

Here, we’ve talk about GNNs where the computation only occurs at the nodes.
        More recent GNN models
        such as Message-Passing Neural Networks and Graph Networks perform computation over the edges as well;
        they compute edge embeddings together with node embeddings.
        This is an even more general framework -
        but the same ‘message passing’ ideas from this section apply.

## Interactive Graph Neural Networks

Below is an interactive visualization of these GNN models on small graphs.
        For clarity, the node features are just real numbers here, shown inside the squares next to each node,
        but the same equations hold when the node features are vectors.

          Choose a GNN model using the tabs at the top. Click on a node to see the update equation at that node for the next iteration.
          Use the sliders on the left to change the weights for the current iteration, and watch how the update equation changes. 
        

In practice, each iteration above is generally thought of as a single ‘neural network layer’.
        This ideology is followed by many popular Graph Neural Network libraries, For example: PyTorch Geometric and StellarGraph . allowing one to compose different types of graph convolutions in the same model.

## From Local to Global Convolutions

The methods we’ve seen so far perform ‘local’ convolutions:
        every node’s feature is updated using a function of its local neighbours’ features.

While performing enough steps of message-passing will eventually ensure that
        information from all nodes in the graph is passed,
        one may wonder if there are more direct ways to perform ‘global’ convolutions.

The answer is yes; we will now describe an approach that was actually first put forward
        in the context of neural networks by ,
        much before any of the GNN models we looked at above.

### Spectral Convolutions

As before, we will focus on the case where nodes have one-dimensional features.
        After choosing an arbitrary node-order, we can stack all of the node features to get a
        ‘feature vector’ x ∈ R n x \in \mathbb{R}^n x ∈ R n .

Key Idea: Given a feature vector x x x , 
        the Laplacian L L L allows us to quantify how smooth x x x is, with respect to G G G .

How?

After normalizing x x x such that ∑ i = 1 n x i 2 = 1 \sum_{i = 1}^n x_i^2 = 1 ∑ i = 1 n ​ x i 2 ​ = 1 ,
        if we look at the following quantity involving L L L : R L R_L R L ​ is formally called the Rayleigh quotient . R L ( x ) = x T L x x T x = ∑ ( i , j ) ∈ E ( x i − x j ) 2 ∑ i x i 2 = ∑ ( i , j ) ∈ E ( x i − x j ) 2 . R_L(x) = \frac{x^T L x}{x^T x} = \frac{\sum_{(i, j) \in E} (x_i - x_j)^2}{\sum_i x_i^2} = \sum_{(i, j) \in E} (x_i - x_j)^2. R L ​ ( x ) = x T x x T L x ​ = ∑ i ​ x i 2 ​ ∑ ( i , j ) ∈ E ​ ( x i ​ − x j ​ ) 2 ​ = ( i , j ) ∈ E ∑ ​ ( x i ​ − x j ​ ) 2 . we immediately see that feature vectors x x x that assign similar values to 
        adjacent nodes in G G G (hence, are smooth) would have smaller values of R L ( x ) R_L(x) R L ​ ( x ) .

L L L is a real, symmetric matrix, which means it has all real eigenvalues λ 1 ≤ … ≤ λ n \lambda_1 \leq \ldots \leq \lambda_{n} λ 1 ​ ≤ … ≤ λ n ​ . An eigenvalue λ \lambda λ of a matrix A A A is a value
          satisfying the equation A u = λ u Au = \lambda u A u = λ u for a certain vector u u u , called an eigenvector.
          For a friendly introduction to eigenvectors,
          please see this tutorial . Further, the corresponding eigenvectors u 1 , … , u n u_1, \ldots, u_{n} u 1 ​ , … , u n ​ can be taken to be orthonormal: u k 1 T u k 2 = { 1 if k 1 = k 2 . 0 if k 1 ≠ k 2 . u_{k_1}^T u_{k_2} =
          \begin{cases}
            1 \quad \text{ if } {k_1} = {k_2}. \\
            0 \quad \text{ if } {k_1} \neq {k_2}.
          \end{cases} u k 1 ​ T ​ u k 2 ​ ​ = { 1 if k 1 ​ = k 2 ​ . 0 if k 1 ​ ≠ k 2 ​ . ​ It turns out that these eigenvectors of L L L are successively less smooth, as R L R_L R L ​ indicates: This is the min-max theorem for eigenvalues. argmin x , x ⊥ { u 1 , … , u i − 1 } R L ( x ) = u i . min x , x ⊥ { u 1 , … , u i − 1 } R L ( x ) = λ i . \underset{x, \ x \perp \{u_1, \ldots, u_{i - 1}\}}{\text{argmin}} R_L(x) = u_i.
          \qquad
          \qquad
          \qquad
          \min_{x, \ x \perp \{u_1, \ldots, u_{i - 1}\}} R_L(x) = \lambda_i. x , x ⊥ { u 1 ​ , … , u i − 1 ​ } argmin ​ R L ​ ( x ) = u i ​ . x , x ⊥ { u 1 ​ , … , u i − 1 ​ } min ​ R L ​ ( x ) = λ i ​ . The set of eigenvalues of L L L are called its ‘spectrum’, hence the name!
        We denote the ‘spectral’ decomposition of L L L as: L = U Λ U T . L = U \Lambda U^T. L = U Λ U T . where Λ \Lambda Λ is the diagonal matrix of sorted eigenvalues,
        and U U U denotes the matrix of the eigenvectors (sorted corresponding to increasing eigenvalues): Λ = [ λ 1 ⋱ λ n ] U = [ u 1 ⋯ u n ] . \Lambda = \begin{bmatrix}
                        \lambda_{1} &        & \\
                                    & \ddots & \\
                                    &        & \lambda_{n}
                    \end{bmatrix}
          \qquad
          \qquad
          \qquad
          \qquad
          U = \begin{bmatrix}  \\ u_1 \ \cdots \ u_n \\ \end{bmatrix}. Λ = ⎣ ⎡ ​ λ 1 ​ ​ ⋱ ​ λ n ​ ​ ⎦ ⎤ ​ U = ⎣ ⎡ ​ u 1 ​ ⋯ u n ​ ​ ⎦ ⎤ ​ . The orthonormality condition between eigenvectors gives us that U T U = I U^T U = I U T U = I , the identity matrix.
        As these n n n eigenvectors form a basis for R n \mathbb{R}^n R n ,
        any feature vector x x x can be represented as a linear combination of these eigenvectors: x = ∑ i = 1 n x i ^ u i = U x ^ . x = \sum_{i = 1}^n \hat{x_i} u_i = U \hat{x}. x = i = 1 ∑ n ​ x i ​ ^ ​ u i ​ = U x ^ . where x ^ \hat{x} x ^ is the vector of coefficients [ x 0 , … x n ] [x_0, \ldots x_n] [ x 0 ​ , … x n ​ ] .
        We call x ^ \hat{x} x ^ as the spectral representation of the feature vector x x x .
        The orthonormality condition allows us to state: x = U x ^ ⟺ U T x = x ^ . x = U \hat{x} \quad \Longleftrightarrow \quad U^T x = \hat{x}. x = U x ^ ⟺ U T x = x ^ . This pair of equations allows us to interconvert
        between the ‘natural’ representation x x x and the ‘spectral’ representation x ^ \hat{x} x ^ for any vector x ∈ R n x \in \mathbb{R}^n x ∈ R n .

### Spectral Representations of Natural Images

As discussed before, we can consider any image as a grid graph, where each pixel is a node,
        connected by edges to adjacent pixels.
        Thus, a pixel can have either 3 , 5 , 3, 5, 3 , 5 , or 8 8 8 neighbours, depending on its location within the image grid.
        Each pixel gets a value as part of the image. If the image is grayscale, each value will be a single 
        real number indicating how dark the pixel is. If the image is colored, each value will be a 3 3 3 -dimensional
        vector, indicating the values for the red, green and blue (RGB) channels. We use the alpha channel as well in the visualization below, so this is actually RGBA.

This construction allows us to compute the graph Laplacian and the eigenvector matrix U U U .
        Given an image, we can then investigate what its spectral representation looks like.

To shed some light on what the spectral representation actually encodes,
        we perform the following experiment over each channel of the image independently:

- We first collect all pixel values across a channel into a feature vector x x x .

- Then, we obtain its spectral representation x ^ \hat{x} x ^ . x ^ = U T x \hat{x} = U^T x x ^ = U T x

- We truncate this to the first m m m components to get x ^ m \hat{x}_m x ^ m ​ .
            By truncation, we mean zeroing out all of the remaining n − m n - m n − m components of x ^ \hat{x} x ^ .
            This truncation is equivalent to using only the first m m m eigenvectors to compute the spectral representation. x ^ m = Truncate m ( x ^ ) \hat{x}_m = \text{Truncate}_m(\hat{x}) x ^ m ​ = Truncate m ​ ( x ^ )

- Then, we convert this truncated representation x ^ m \hat{x}_m x ^ m ​ back to the natural basis to get x m x_m x m ​ . x m = U x ^ m x_m = U \hat{x}_m x m ​ = U x ^ m ​

Finally, we stack the resulting channels back together to get back an image.
        We can now see how the resulting image changes with choices of m m m .
        Note that when m = n m = n m = n , the resulting image is identical to the original image,
        as we can reconstruct each channel exactly.

          Use the radio buttons at the top to chose one of the four sample images.
          Each of these images has been taken from the ImageNet 
          dataset and downsampled to 505050 pixels wide and 404040 pixels tall.
          As there are n=50×40=2000n = 50 \times 40 = 2000n=50×40=2000 pixels in each image, there are 200020002000 Laplacian eigenvectors.
          Use the slider at the bottom to change the number of spectral components to keep, noting how
          images get progressively blurrier as the number of components decrease.
        

As m m m decreases, we see that the output image x m x_m x m ​ gets blurrier.
        If we decrease m m m to 1 1 1 , the output image x m x_m x m ​ is entirely the same color throughout.
        We see that we do not need to keep all n n n components;
        we can retain a lot of the information in the image with significantly fewer components.

        We can relate this to the Fourier decomposition of images:
        the more eigenvectors we use, the higher frequencies we can represent on the grid.

To complement the visualization above,
        we additionally visualize the first few eigenvectors on a smaller 8 × 8 8 \times 8 8 × 8 grid below.
        We change the coefficients of the first 1 0 10 1 0 out of 6 4 64 6 4 eigenvectors
        in the spectral representation
        and see how the resulting image changes:

          Move the sliders to change the spectral representation x^\hat{x}x^ (right),
          and see how xxx itself changes on the image (left).
          Note how the first eigenvectors are much ‘smoother’ than the later ones,
          and the many patterns we can make with only 101010 eigenvectors.
        

These visualizations should convince you that the first eigenvectors are indeed smooth,
        and the smoothness correspondingly decreases as we consider later eigenvectors.

For any image x x x , we can think of
        the initial entries of the spectral representation x ^ \hat{x} x ^ as capturing ‘global’ image-wide trends, which are the low-frequency components,
        while the later entries as capturing ‘local’ details, which are the high-frequency components.

### Embedding Computation

We now have the background to understand spectral convolutions
        and how they can be used to compute embeddings/feature representations of nodes.

As before, the model we describe below has K K K layers:
        each layer k k k has learnable parameters w ^ ( k ) \hat{w}^{(k)} w ^ ( k ) ,
        called the ‘filter weights’.
        These weights will be convolved with the spectral representations of the node features.
        As a result, the number of weights needed in each layer is equal to m m m , the number of 
        eigenvectors used to compute the spectral representations.
        We had shown in the previous section that we can take m ≪ n m \ll n m ≪ n and still not lose out on significant amounts of information.

Thus, convolution in the spectral domain enables the use of significantly fewer parameters
        than just direct convolution in the natural domain.
        Further, by virtue of the smoothness of the Laplacian eigenvectors across the graph,
        using spectral representations automatically enforces an inductive bias for
        neighbouring nodes to get similar representations.

Assuming one-dimensional node features for now,
        the output of each layer is a vector of node representations h ( k ) h^{(k)} h ( k ) ,
        where each node’s representation corresponds to a row
        of the vector.

We fix an ordering of the nodes in G G G . This gives us the adjacency matrix A A A and the graph Laplacian L L L ,
        allowing us to compute U m U_m U m ​ .
        Finally, we can describe the computation that the layers perform, one after the other:

The method above generalizes easily to the case where each h ( k ) ∈ R d k h^{(k)} \in \mathbb{R}^{d_k} h ( k ) ∈ R d k ​ , as well:
        see for details.

With the insights from the previous section, we see that convolution in the spectral-domain of graphs
        can be thought of as the generalization of convolution in the frequency-domain of images.

### Spectral Convolutions are Node-Order Equivariant

We can show spectral convolutions are node-order equivariant using a similar approach
        as for Laplacian polynomial filters.

          Details for the Interested Reader
        

As in our proof before ,
          let’s fix an arbitrary node-order.
          Then, any other node-order can be represented by a
          permutation of this original node-order.
          We can associate this permutation with its permutation matrix P P P .

          Under this new node-order,
          the quantities below transform in the following way: x → P x A → P A P T L → P L P T U m → P U m \begin{aligned}
              x &\to Px \\
              A &\to PAP^T \\
              L &\to PLP^T \\
              U_m &\to PU_m
            \end{aligned} x A L U m ​ ​ → P x → P A P T → P L P T → P U m ​ ​ which implies that, in the embedding computation: x ^ → ( P U m ) T ( P x ) = U m T x = x ^ w ^ → ( P U m ) T ( P w ) = U m T w = w ^ g ^ → g ^ g → ( P U m ) g ^ = P ( U m g ^ ) = P g \begin{aligned}
              \hat{x} &\to \left(PU_m\right)^T (Px) = U_m^T x = \hat{x} \\
              \hat{w} &\to \left(PU_m\right)^T (Pw) = U_m^T w = \hat{w} \\
              \hat{g} &\to \hat{g} \\
              g &\to (PU_m)\hat{g} = P(U_m\hat{g}) = Pg
            \end{aligned} x ^ w ^ g ^ ​ g ​ → ( P U m ​ ) T ( P x ) = U m T ​ x = x ^ → ( P U m ​ ) T ( P w ) = U m T ​ w = w ^ → g ^ ​ → ( P U m ​ ) g ^ ​ = P ( U m ​ g ^ ​ ) = P g ​ Hence, as σ \sigma σ is applied elementwise: f ( P x ) = σ ( P g ) = P σ ( g ) = P f ( x ) f(Px) = \sigma(Pg) = P \sigma(g) = P f(x) f ( P x ) = σ ( P g ) = P σ ( g ) = P f ( x ) as required.
          Further, we see that the spectral quantities x ^ , w ^ \hat{x}, \hat{w} x ^ , w ^ and g ^ \hat{g} g ^ ​ are unchanged by permutations of the nodes. Formally, they are what we would call node-order invariant.

The theory of spectral convolutions is mathematically well-grounded;
        however, there are some key disadvantages that we must talk about:

- We need to compute the eigenvector matrix U m U_m U m ​ from L L L . For large graphs, this becomes quite infeasible.

- Even if we can compute U m U_m U m ​ , global convolutions themselves are inefficient to compute,
          because of the repeated
          multiplications with U m U_m U m ​ and U m T U_m^T U m T ​ .

- The learned filters are specific to the input graphs,
          as they are represented in terms
          of the spectral decomposition of input graph Laplacian L L L .
          This means they do not transfer well to new graphs
          which have significantly different structure (and hence, significantly
          different eigenvalues) .

While spectral convolutions have largely been superseded by
        ‘local’ convolutions for the reasons discussed above,
        there is still much merit to understanding the ideas behind them.
        Indeed, a recently proposed GNN model called Directional Graph Networks actually uses the Laplacian eigenvectors
        and their mathematical properties
        extensively.

### Global Propagation via Graph Embeddings

A simpler way to incorporate graph-level information
        is to compute embeddings of the entire graph by pooling node
        (and possibly edge) embeddings,
        and then using the graph embedding to update node embeddings,
        following an iterative scheme similar to what we have looked at here.
        This is an approach used by Graph Networks .
        We will briefly discuss how graph-level embeddings
        can be constructed in Pooling .
        However, such approaches tend to ignore the underlying
        topology of the graph that spectral convolutions can capture.

## Learning GNN Parameters

All of the embedding computations we’ve described here, whether spectral or spatial, are completely differentiable.
        This allows GNNs to be trained in an end-to-end fashion, just like a standard neural network,
        once a suitable loss function L \mathcal{L} L is defined:

- Node Classification : By minimizing any of the standard losses for classification tasks,
            such as categorical cross-entropy when multiple classes are present: L ( y v , y v ^ ) = − ∑ c y v c log y v c ^ . \mathcal{L}(y_v, \hat{y_v}) = -\sum_{c} y_{vc} \log{\hat{y_{vc}}}. L ( y v ​ , y v ​ ^ ​ ) = − c ∑ ​ y v c ​ lo g y v c ​ ^ ​ . where y v c ^ \hat{y_{vc}} y v c ​ ^ ​ is the predicted probability that node v v v is in class c c c .
            GNNs adapt well to the semi-supervised setting, which is when only some nodes in the graph are labelled.
            In this setting, one way to define a loss L G \mathcal{L}_{G} L G ​ over an input graph G G G is: L G = ∑ v ∈ Lab ( G ) L ( y v , y v ^ ) ∣ Lab ( G ) ∣ \mathcal{L}_{G} = \frac{\sum\limits_{v \in \text{Lab}(G)} \mathcal{L}(y_v, \hat{y_v})}{| \text{Lab}(G) |} L G ​ = ∣ Lab ( G ) ∣ v ∈ Lab ( G ) ∑ ​ L ( y v ​ , y v ​ ^ ​ ) ​ where, we only compute losses over labelled nodes Lab ( G ) \text{Lab}(G) Lab ( G ) .

- Graph Classification : By aggregating node representations,
            one can construct a vector representation of the entire graph.
            This graph representation can be used for any graph-level task, even beyond classification.
            See Pooling for how representations of graphs can be constructed.

- Link Prediction : By sampling pairs of adjacent and non-adjacent nodes,
            and use these vector pairs as inputs to predict the presence/absence of an edge.
            For a concrete example, by minimizing the following ‘logistic regression’-like loss: L ( y v , y u , e v u ) = − e v u log ( p v u ) − ( 1 − e v u ) log ( 1 − p v u ) p v u = σ ( y v T y u ) \begin{aligned}
              \mathcal{L}(y_v, y_u, e_{vu}) &= -e_{vu} \log(p_{vu}) - (1 - e_{vu}) \log(1 - p_{vu}) \\
              p_{vu} &= \sigma(y_v^Ty_u)
              \end{aligned} L ( y v ​ , y u ​ , e v u ​ ) p v u ​ ​ = − e v u ​ lo g ( p v u ​ ) − ( 1 − e v u ​ ) lo g ( 1 − p v u ​ ) = σ ( y v T ​ y u ​ ) ​ where σ \sigma σ is the sigmoid function ,
            and e v u = 1 e_{vu} = 1 e v u ​ = 1 iff there is an edge between nodes v v v and u u u , being 0 0 0 otherwise.

- Node Clustering : By simply clustering the learned node representations.

The broad success of pre-training for natural language processing models
        such as ELMo and BERT has sparked interest in similar techniques for GNNs .
        The key idea in each of these papers is to train GNNs to predict
        local (eg. node degrees, clustering coefficient, masked node attributes)
        and/or global graph properties (eg. pairwise distances, masked global attributes).

Another self-supervised technique is to enforce that neighbouring nodes get similar embeddings,
        mimicking random-walk approaches such as node2vec and DeepWalk :

LG=∑v∑u∈NR(v)logexpzvTzu∑u′expzu′Tzu.
          L_{G} = \sum_{v} \sum_{u \in N_R(v)} \log\frac{\exp{z_v^T z_u}}{\sum\limits_{u’} \exp{z_{u’}^T z_u}}.
        LG​=v∑​u∈NR​(v)∑​logu′∑​expzu′T​zu​expzvT​zu​​.

where N R ( v ) N_R(v) N R ​ ( v ) is a multi-set of nodes visited when random walks are started from v v v .
        For large graphs, where computing the sum over all nodes may be computationally expensive,
        techniques such as Noise Contrastive Estimation are especially useful.

## Conclusion and Further Reading

While we have looked at many techniques and ideas in this article,
        the field of Graph Neural Networks is extremely vast.
        We have been forced to restrict our discussion to a small subset of the entire literature,
        while still communicating the key ideas and design principles behind GNNs.
        We recommend the interested reader take a look at for a more comprehensive survey.

We end with pointers and references for additional concepts readers might be interested in:

### GNNs in Practice

It turns out that accomodating the different structures of graphs is often hard to do efficiently,
        but we can still represent many GNN update equations using
        as sparse matrix-vector products (since generally, the adjacency matrix is sparse for most real-world graph datasets.)
        For example, the GCN variant discussed here can be represented as: h ( k ) = D − 1 A ⋅ h ( k − 1 ) W ( k ) T + h ( k − 1 ) B ( k ) T . h^{(k)} = D^{-1} A \cdot h^{(k - 1)} {W^{(k)}}^T + h^{(k - 1)} {B^{(k)}}^T. h ( k ) = D − 1 A ⋅ h ( k − 1 ) W ( k ) T + h ( k − 1 ) B ( k ) T . Restructuring the update equations in this way allows for efficient vectorized implementations of GNNs on accelerators
        such as GPUs.

Regularization techniques for standard neural networks,
        such as Dropout ,
        can be applied in a straightforward manner to the parameters
        (for example, zero out entire rows of W ( k ) W^{(k)} W ( k ) above).
        However, there are graph-specific techniques such as DropEdge that removes entire edges at random from the graph,
        that also boost the performance of many GNN models.

### Different Kinds of Graphs

Here, we have focused on undirected graphs, to avoid going into too many unnecessary details.
        However, there are some simple variants of spatial convolutions for:

- Directed graphs: Aggregate across in-neighbourhood and/or out-neighbourhood features.

- Temporal graphs: Aggregate across previous and/or future node features.

- Heterogeneous graphs: Learn different aggregation functions for each node/edge type.

There do exist more sophisticated techniques that can take advantage of the different structures of these graphs:
        see for more discussion.

### Pooling

This article discusses how GNNs compute useful representations of nodes.
          But what if we wanted to compute representations of graphs for graph-level tasks (for example, predicting the toxicity of a molecule)?

A simple solution is to just aggregate the final node embeddings and pass them through another neural network PREDICT G \text{PREDICT}_G PREDICT G ​ : h G = PREDICT G ( AGG v ∈ G ( { h v } ) ) h_G = \text{PREDICT}_G \Big( \text{AGG}_{v \in G}\left(\{ h_v \} \right) \Big) h G ​ = PREDICT G ​ ( AGG v ∈ G ​ ( { h v ​ } ) ) However, there do exist more powerful techniques for ‘pooling’ together node representations:

- SortPool : Sort vertices of the graph to get a fixed-size node-order invariant representation of the graph, and then apply any standard neural network architecture.

- DiffPool : Learn to cluster vertices, build a coarser graph over clusters instead of nodes, then apply a GNN over the coarser graph. Repeat until only one cluster is left.

- SAGPool : Apply a GNN to learn node scores, then keep only the nodes with the top scores, throwing away the rest. Repeat until only one node is left.

## Supplementary Material

### Reproducing Experiments

The experiments from Spectral Representations of Natural Images can be reproduced using the following
        Colab notebook: Spectral Representations of Natural Images .

### Recreating Visualizations

To aid in the creation of future interactive articles,
        we have created ObservableHQ notebooks for each of the interactive visualizations here:
