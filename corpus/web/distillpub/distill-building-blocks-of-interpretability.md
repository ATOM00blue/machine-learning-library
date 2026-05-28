---
title: "The Building Blocks of Interpretability"
source: "web"
url: "https://distill.pub/2018/building-blocks/"
domain: "distill.pub"
name: "Distill - Building Blocks of Interpretability"
fetched_at: "2026-05-26T23:30:20Z"
topics: ["interpretability"]
---

With the growing success of neural networks, there is a corresponding need to be able to explain their decisions — including building confidence about how they will behave in the real-world, detecting model bias, and for scientific curiosity.

    In order to do so, we need to both construct deep abstractions and reify (or instantiate) them in rich interfaces .

    With a few exceptions , existing work on interpretability fails to do these in concert.

The machine learning community has primarily focused on developing powerful methods, such as feature visualization , attribution , and dimensionality reduction , for reasoning about neural networks. 

    However, these techniques have been studied as isolated threads of research, and the corresponding work of reifying them has been neglected.

    On the other hand, the human-computer interaction community has begun to explore rich user interfaces for neural networks , but they have not yet engaged deeply with these abstractions. 

    To the extent these abstractions have been used, it has been in fairly standard ways.

    As a result, we have been left with impoverished interfaces (e.g., saliency maps or correlating abstract neurons) that leave a lot of value on the table. 

    Worse, many interpretability techniques have not been fully actualized into abstractions because there has not been pressure to make them generalizable or composable.

In this article, we treat existing interpretability methods as fundamental and composable building blocks for rich user interfaces.

    We find that these disparate techniques now come together in a unified grammar, fulfilling complementary roles in the resulting interfaces.

    Moreover, this grammar allows us to systematically explore the space of interpretability interfaces, enabling us to evaluate whether they meet particular goals.

    We will present interfaces that show what the network detects and explain how it develops its understanding, while keeping the amount of information human-scale .

    For example, we will see how a network looking at a labrador retriever detects floppy ears and how that influences its classification.

    Our interfaces are speculative and one might wonder how reliable they are. 

    Rather than address this point piecemeal, we dedicate a section to it at the end of the article.
  

In this article, we use GoogLeNet , an image classification model, to demonstrate our interface ideas because its neurons seem unusually semantically meaningful. We’re actively investigating why this is, and hope to uncover principles for designing interpretable models. In the meantime, while we demonstrate our techniques on GoogLeNet, we provide code for you to try them on other models. Although here we’ve made a specific choice of task and network, the basic abstractions and patterns for combining them that we present can be applied to neural networks in other domains.

## Making Sense of Hidden Layers

Much of the recent work on interpretability is concerned with a neural network’s input and output layers.
    Arguably, this focus is due to the clear meaning these layers have: in computer vision, the input layer represents values for the red, green, and blue color channels for every pixel in the input image, while the output layer consists of class labels and their associated probabilities.

However, the power of neural networks lies in their hidden layers — at every layer, the network discovers a new representation of the input.

    In computer vision, we use neural networks that run the same feature detectors at every position in the image.

    We can think of each layer’s learned representation as a three-dimensional cube. Each cell in the cube is an activation , or the amount a neuron fires.

    The x- and y-axes correspond to positions in the image, and the z-axis is the channel (or detector) being run.

      The cube of activations that a neural network for computer vision develops at each hidden layer. 
      
      Different slices of the cube allow us to target the activations of individual neurons, spatial positions, or channels.
    

To make a semantic dictionary, we pair every neuron activation with a visualization of that neuron and sort them by the magnitude of the activation.

    This marriage of activations and feature visualization changes our relationship with the underlying mathematical object.

    Activations now map to iconic representations, instead of abstract indices, with many appearing to be similar to salient human ideas, such as “floppy ear,” “dog snout,” or “fur.”

    We use optimization-based feature visualization to avoid spurious correlation, but one could use other methods.
  

Semantic dictionaries are powerful not just because they move away from meaningless indices, but because they express a neural network’s learned abstractions with canonical examples.
        
    With image classification, the neural network learns a set of visual abstractions and thus images are the most natural symbols to represent them.

    Were we working with audio, the more natural symbols would most likely be audio clips.

    This is important because when neurons appear to correspond to human ideas, it is tempting to reduce them to words.

    Doing so, however, is a lossy operation — even for familiar abstractions, the network may have learned a deeper nuance.

    For instance, GoogLeNet has multiple floppy ear detectors that appear to detect slightly different levels of droopiness, length, and surrounding context to the ears.

    There also may exist abstractions which are visually familiar, yet that we lack good natural language descriptions for: for example, take the particular column of shimmering light where sun hits rippling water.

    Moreover, the network may learn new abstractions that appear alien to us — here, natural language would fail us entirely!

    In general, canonical examples are a more natural way to represent the foreign abstractions that neural networks learn than native human language.

By bringing meaning to hidden layers, semantic dictionaries set the stage for our existing interpretability techniques to be composable building blocks.

    As we shall see, just like their underlying vectors, we can apply dimensionality reduction to them.

    In other cases, semantic dictionaries allow us to push these techniques further.

    For example, besides the one-way attribution that we currently perform with the input and output layers, semantic dictionaries allow us to attribute to-and-from specific hidden layers.

    In principle, this work could have been done without semantic dictionaries but it would have been unclear what the results meant.

    While we introduce semantic dictionaries in terms of neurons, they can be used with any basis of activations. We will explore
    this more later.
  

## What Does the Network See?

Applying this technique to all the activation vectors allows us to not only see what the network detects at each position, but also what the network understands of the input image as a whole.

And, by working across layers (eg. “mixed3a”, “mixed4d”), we can observe how the network’s understanding evolves: from detecting edges in earlier layers, to more sophisticated shapes and object parts in the latter.

These visualizations, however, omit a crucial piece of information: the magnitude of the activations.
    
    By scaling the area of each cell by the magnitude of the activation vector, we can indicate how strongly the network detected features at that position:

## How Are Concepts Assembled?

Feature visualization helps us answer what the network detects, but it does not answer how the network assembles these individual pieces to arrive at later decisions, or why these decisions were made.

Attribution is a set of techniques that answers such questions by explaining the relationships between neurons.

    There are a wide variety of approaches to attribution but, so far, there doesn’t seem to be a clear right answer.

    In fact, there’s reason to think that all our present answers aren’t quite right .

    We think there’s a lot of important research to be done on attribution methods, but for the purposes of this article the exact approach taken to attribution doesn’t matter.

    We use a fairly simple method, linearly approximating the relationship We do attribution by linear approximation in all of our interfaces. That is, we estimate the effect of a neuron on the output is its activation times the rate at which increasing its activation increases the output. When we talk about a linear combination of activations, the attribution can be thought of as the linear combination of the attributions of the units, or equivalently as the dot product between the activation of that combination and the gradient. For spatial attribution, we do an additional trick. GoogLeNet’s strided max pooling introduces a lot of noise and checkerboard patterns to it’s gradients . To avoid our interface demonstrations being dominated by this noise, we (a) do a relaxation of the gradient of max pooling, distributing gradient to inputs proportional to their activation instead of winner takes all and (b) cancel out the checkerboard patterns. The notebooks attached to diagrams provide reference implementations. , but could easily substitute in essentially any other technique.

    Future improvements to attribution will, of course, correspondingly improve the interfaces built on top of them.

### Spatial Attribution with Saliency Maps

The most common interface for attribution is called a saliency map — a simple heatmap that highlights pixels of the input image that most caused the output classification.

    We see two weaknesses with this current approach.

First, it is not clear that individual pixels should be the primary unit of attribution.

    The meaning of each pixel is extremely entangled with other pixels, is not robust to simple visual transforms (e.g., brightness, contrast, etc.), and is far-removed from high-level concepts like the output class.

    Second, traditional saliency maps are a very limited type of interface — they only display the attribution for a single class at a time, and do not allow you to probe into individual points more deeply.

    As they do not explicitly deal with hidden layers, it has been difficult to fully explore their design space.

We instead treat attribution as another user interface building block, and apply it to the hidden layers of a neural network. 

    In doing so, we change the questions we can pose.

    Rather than asking whether the color of a particular pixel was important for the “labrador retriever” classification, we instead ask whether the high-level idea detected at that position (such as “floppy ear”) was important.

    This approach is similar to what Class Activation Mapping (CAM) methods do but, because they interpret their results back onto the input image, they miss the opportunity to communicate in terms of the rich behavior of a network’s hidden layers.

The above interface affords us a more flexible relationship with attribution.

    To start, we perform attribution from each spatial position of each hidden layer shown to all 1,000 output classes.

    In order to visualize this thousand-dimensional vector, we use dimensionality reduction to produce a multi-directional saliency map.

    Overlaying these saliency maps on our magnitude-sized activation grids provides an information scent over attribution space.

    The activation grids allow us to anchor attribution to the visual vocabulary our semantic dictionaries first established.

    On hover, we update the legend to depict attribution to the output classes (i.e., which classes does this spatial position most contribute to?).

Perhaps most interestingly, this interface allows us to interactively perform attribution between hidden layers .

    On hover, additional saliency maps mask the hidden layers, in a sense shining a light into their black boxes.

    This type of layer-to-layer attribution is a prime example of how carefully considering interface design drives the generalization of our existing abstractions for interpretability.

With this diagram, we have begun to think of attribution in terms of higher-level concepts.

    However, at a particular position, many concepts are being detected together and this interface makes it difﬁcult to split them apart. 
    
    By continuing to focus on spatial positions, these concepts remain entangled.

### Channel Attribution

Saliency maps implicitly slice our cube of activations by applying attribution to the spatial positions of a hidden layer.

    This aggregates over all channels and, as a result, we cannot tell which specific detectors at each position most contributed to the final output classification.

An alternate way to slice the cube is by channels instead of spatial locations.

    Doing so allows us to perform channel attribution : how much did each detector contribute to the final output?

    (This approach is similar to contemporaneous work by Kim et al. , who do attribution to learned combination of channels.)

This diagram is analogous to the previous one we saw: we conduct layer-to-layer attribution but this time over channels rather than spatial positions.

    Once again, we use the icons from our semantic dictionary to represent the channels that most contribute to the final output classification.

    Hovering over an individual channel displays a heatmap of its activations overlaid on the input image.

    The legend also updates to show its attribution to the output classes (i.e., what are the top classes this channel supports?).

    Clicking a channel allows us to drill into the layer-to-layer attributions, identifying the channels at lower layers that most contributed as well as the channels at higher layers that are most supported.

While these diagrams focus on layer-to-layer attribution, it can still be valuable to focus on a single hidden layer.

    For example, the teaser figure allows us to evaluate hypotheses for why one class succeeded over the other.

Attribution to spatial locations and channels can reveal powerful things about a model, especially when we combine them together.

    Unfortunately, this family of approaches is burdened by two significant problems.

    On the one hand, it is very easy to end up with an overwhelming amount of information: it would take hours of human auditing to understand the long-tail of channels that slightly impact the output.

    On the other hand, both the aggregations we have explored are extremely lossy and can miss important parts of the story.

    And, while we could avoid lossy aggregation by working with individual neurons, and not aggregating at all, this explodes the first problem combinatorially.

## Making Things Human-Scale

In previous sections, we’ve considered three ways of slicing the cube of activations: into spatial activations, channels, and individual neurons.
    Each of these has major downsides.
    If one only uses spatial activations or channels, they miss out on very important parts of the story.
    For example it’s interesting that the floppy ear detector helped us classify an image as a Labrador retriever, but it’s much more interesting when that’s combined with the locations that fired to do so.
    One can try to drill down to the level of neurons to tell the whole story, but the tens of thousands of neurons are simply too much information.
    Even the hundreds of channels, before being split into individual neurons, can be overwhelming to show users!

If we want to make useful interfaces into neural networks, it isn’t enough to make things meaningful.
    We need to make them human scale, rather than overwhelming dumps of information.
    The key to doing so is finding more meaningful ways of breaking up our activations.
    There is good reason to believe that such decompositions exist.
    Often, many channels or spatial positions will work together in a highly correlated way and are most useful to think of as one unit.
    Other channels or positions will have very little activity, and can be ignored for a high-level overview.
    So, it seems like we ought to be able to find better decompositions if we had the right tools.

There is an entire field of research, called matrix factorization, that studies optimal strategies for breaking up matrices.
    By flattening our cube into a matrix of spatial locations and channels, we can apply these techniques to get more meaningful groups of neurons.
    These groups will not align as naturally with the cube as the groupings we previously looked at.
    Instead, they will be combinations of spatial locations and channels.
    Moreover, these groups are constructed to explain the behavior of a network on a particular image.
    It would not be effective to reuse the same groupings on another image; each image requires calculating a unique set of groups.

      In addition to naturally slicing a hidden layer’s cube of activations into neurons, spatial locations, or channels, we can also consider more arbitrary groupings of locations and channels.
    

The groups that come out of this factorization will be the atoms of the interface a user works with. Unfortunately, any grouping is inherently a tradeoff between reducing things to human scale and, because any aggregation is lossy, preserving information. Matrix factorization lets us pick what our groupings are optimized for, giving us a better tradeoff than the natural groupings we saw earlier.

The goals of our user interface should influence what we optimize our matrix factorization to prioritize. For example, if we want to prioritize what the network detected, we would want the factorization to fully describe the activations. If we instead wanted to prioritize what would change the network’s behavior, we would want the factorization to fully describe the gradient. Finally, if we want to prioritize what caused the present behavior, we would want the factorization to fully describe the attributions. Of course, we can strike a balance between these three objectives rather than optimizing one to the exclusion of the others.

In the following diagram, we’ve constructed groups that prioritize the activations, by factorizing the activations Most matrix factorization algorithms and libraries are set up to minimize the mean squared error of the reconstruction of a matrix you give them. There are ways to hack such libraries to achieve more general objectives through clever manipulations of the provided matrix, as we will see below. More broadly, matrix factorization is an optimization problem, and with custom tools you can achieve all sorts of custom factorizations. with non-negative matrix factorization As the name suggests, non-negative matrix factorization (NMF) constrains its factors to be positive. This is fine for the activations of a ReLU network, which must be positive as well. Our experience is that the groups we get from NMF seem more independent and semantically meaningful than those without this constraint. Because of this constraints, groups from NMF are a less efficient at representing the activations than they would be without, but our experience is that they seem more independent and semantically meaningful. .
    Notice how the overwhelmingly large number of neurons has been reduced to a small set of groups, concisely summarizing the story of the neural network.

This figure only focuses at a single layer but, as we saw earlier, it can be useful to look across multiple layers to understand how a neural network assembles together lower-level detectors into higher-level concepts.

The groups we constructed before were optimized to understand a single layer independent of the others. To understand multiple layers together, we would like each layer’s factorization to be “compatible” — to have the groups of earlier layers naturally compose into the groups of later layers. This is also something we can optimize the factorization for We formalize this “compatibility” in a manner described below, although we’re not confident it’s the best formalization and won’t be surprised if it is superseded in future work. Consider the attribution from every neuron in the layer to the set of N groups we want it to be compatible with.
        The basic idea is to split each entry in the activation matrix into N entries on the channel dimension, spreading the values proportional to the absolute value of its attribution to the corresponding group.
        Any factorization of this matrix induces a factorization of the original matrix by collapsing the duplicated entries in the column factors.
        However, the resulting factorization tries to create separate factors when the activation of the same channel has different attributions in different places. .

In this section, we recognize that the way in which we break apart the cube of activations is an important interface decision. Rather than resigning ourselves to the natural slices of the cube of activations, we construct more optimal groupings of neurons. These improved groupings are both more meaningful and more human-scale, making it less tedious for users to understand the behavior of the network.

Our visualizations have only begun to explore the potential of alternate bases in providing better atoms for understanding neural networks.
    For example, while we focus on creating smaller numbers of directions to explain individual examples, there’s recently been exciting work finding “globally” meaningful directions — such bases could be especially helpful when trying to understand multiple examples at a time, or in comparing models. The recent NIPS disentangling workshop provides other promising directions. We’re excited to see a venue for this developing area of research.

## The Space of Interpretability Interfaces

The interface ideas presented in this article combine building blocks such as feature visualization and attribution.

    Composing these pieces is not an arbitrary process, but rather follows a structure based on the goals of the interface.

    For example, should the interface emphasize what the network recognizes, prioritize how its understanding develops, or focus on making things human-scale .

    To evaluate such goals, and understand the tradeoffs, we need to be able to systematically consider possible alternatives.

We can think of an interface as a union of individual elements.

Each element displays a specific type of content (e.g., activations or attribution) using a particular style of presentation (e.g., feature visualization or traditional information visualization).

    This content lives on substrates defined by how given layers of the network are broken apart into atoms , and may be transformed by a series of operations (e.g., to filter it or project it onto another substrate).

    For example, our semantic dictionaries use feature visualization to display the activations of a hidden layer's neurons .

One way to represent this way of thinking is with a formal grammar , but we find it helpful to think about the space visually.

    We can represent the network’s substrate (which layers we display, and how we break them apart) as a grid, with the content and style of presentation plotted on this grid as points and connections.

![](images/design_space/empty.svg)

This setup gives us a framework to begin exploring the space of interpretability interfaces step by step.

    For instance, let us consider our teaser figure again.

    Its goal is to help us compare two potential classifications for an input image.

![](images/design_space/teaser.svg)

1. Feature visualization
![](images/teaser-1.png)

To understand a classification, we focus on the channels of the mixed4d layer. Feature visualization makes these channels meaningful.

2. Filter by output attribution
![](images/teaser-2.png)

Next, we filter for specific classes by calculating the output attribution.

3. Drill down on hover
![](images/teaser-3.png)

Hovering over channels, we get a heatmap of spatial activations.

In this article, we have only scratched the surface of possibilities.

    There are lots of combinations of our building blocks left to explore, and the design space gives us a way to do so systematically.

Moreover, each building block represents a broad class of techniques.

    Our interfaces take only one approach but, as we saw in each section, there are a number of alternatives for feature visualization, attribution, and matrix factorization.

    An immediate next step would be to try using these alternate techniques, and research ways to improve them.

Finally, this is not the complete set of building blocks; as new ones are discovered, they expand the space.

    For example, Koh & Liang. suggest ways of understanding the influence of dataset examples on model behavior .

    We can think of dataset examples as another substrate in our design space, thus becoming another building block that fully composes with the others.

    In doing so, we can now imagine interfaces that not only allow us to inspect the influence of dataset examples on the final output classification (as Koh & Liang proposed), but also how examples influence the features of hidden layers, and how they influence the relationship between these features and the output.

    For example, if we consider our “Labrador retriever” image, we can not only see which dataset examples most influenced the model to arrive at this classification, but also which dataset examples most caused the “floppy ear” detectors to fire, and which dataset examples most caused these detectors to increase the “Labrador retriever” classification.

![](images/design_space/dataset.svg)

        A new substrate.
      

        An interface to understand how dataset examples influence the output classification, as presented by Koh & Liang

        An interface showing how examples influence the channels of hidden layers.
      

        An interface for identifying which dataset examples most caused particular detectors to increase the output classification.
      

Beyond interfaces for analyzing model behavior, if we add model parameters as a substrate, the design space now allows us to consider interfaces for taking action on neural networks. Note that essentially all our interpretability techniques are differentiable, so you can backprop through them. While most models today are trained to optimize simple objective functions that one can easily describe, many of the things we’d like models to do in the real world are subtle, nuanced, and hard to describe mathematically. An extreme example of the subtle objective problem is something like “creating interesting art”, but much more mundane examples arise more or less whenever humans are involved. One very promising approach to training models for these subtle objectives is learning from human feedback .
    
    However, even with human feedback, it may still be hard to train models to behave the way we want if the problematic aspect of the model doesn’t surface strongly in the training regime where humans are giving feedback. There are lots of reasons why problematic behavior may not surface or may be hard for an evaluator to give feedback on.

      For example, discrimination and bias may be subtly present throughout the model’s behavior, such that it’s hard for a human evaluator to critique.
      
      Or the model may be making a decision in a way that has problematic consequences, but those consequences never play out in the problems we’re training it on. Human feedback on the model’s decision making process, facilitated by interpretability interfaces, could be a powerful solution to these problems.
    
    It might allow us to train models not just to make the right decisions , but to make them for the right reasons .

    (There is however a danger here: we are optimizing our model to look the way we want in our interface — if we aren’t careful, this may lead to the model fooling us! Related ideas have occasionally been discussed under the term “cognitive steganography.” )

Another exciting possibility is interfaces for comparing multiple models.

    For instance, we might want to see how a model evolves during training, or how it changes when you transfer it to a new task. 

    Or, we might want to understand how a whole family of models compares to each other.

    Existing work has primarily focused on comparing the output behavior of models but more recent work is starting to explore comparing their internal representations as well .

    One of the unique challenges of this work is that we may want to align the atoms of each model; if we have completely different models, can we find the most analogous neurons between them?

    Zooming out, can we develop interfaces that allow us to evaluate large spaces of models at once ?

## How Trustworthy Are These Interfaces?

In order for interpretability interfaces to be effective, we must trust the story they are telling us. 

    We perceive two concerns with the set of building blocks we currently use.

    First, do neurons have a relatively consistent meaning across different input images, and is that meaning accurately reified by feature visualization?

    Semantic dictionaries, and the interfaces that build on top of them, are premised off this question being true.

    Second, does attribution make sense and do we trust any of the attribution methods we presently have?

Much prior research has found that directions in neural networks are semantically meaningful .

    One particularly striking example of this is “semantic arithmetic” (eg. “king” - “man” + “woman” = “queen”) . 

    We explored this question, in depth, for GoogLeNet in our previous article and found that many of its neurons seem to correspond to meaningful ideas. We validated this in a number of ways: we visualized them without a generative model prior, so that the content of the visualizations
    was causally linked to the neuron firing; we inspected the spectrum of examples that cause the neuron to fire; and used diversity
    visualizations to try to create different inputs that cause the neuron to fire. For more details, see the article’s appendix and the guided tour in @ch402′s Twitter thread . We’re actively investigating why GoogLeNet’s neurons seem more meaningful. Besides these neurons, however, we also found many neurons that do not have as clean a meaning including “poly-semantic” neurons that respond to a mixture of salient ideas (e.g., “cat” and “car”).

    There are natural ways that interfaces could respond to this: we could use diversity visualizations to reveal the variety of meanings the neuron can take, or rotate our semantic dictionaries so their components are more disentangled.

    Of course, just like our models can be fooled, the features that make them up can be too — including with adversarial examples .

    In our view, features do not need to be flawless detectors for it to be useful for us to think about them as such. 
    
    In fact, it can be interesting to identify when a detector misfires.

With regards to attribution, recent work suggests that many of our current techniques are unreliable .

    One might even wonder if the idea is fundamentally flawed, since a function’s output could be the result of non-linear interactions between its inputs. 

    One way these interactions can pan out is as attribution being “path-dependent” .

    A natural response to this would be for interfaces to explicitly surface this information: how path-dependent is the attribution?

    A deeper concern, however, would be whether this path-dependency dominates the attribution. 

    Clearly, this is not a concern for attribution between adjacent layers because of the simple (essentially linear) mapping between them. 
    
    While there may be technicalities about correlated inputs, we believe that attribution is on firm grounding here.

    And even with layers further apart, our experience has been that attribution between high-level features at the output is much more consistent than attribution to the input — we believe that path-dependence is not a dominating concern here.

Model behavior is extremely complex, and our current building blocks force us to show only speciﬁc aspects of it. 
    
    An important direction for future interpretability research will be developing techniques that achieve broader coverage of model behavior. 
    
    But, even with such improvements, we anticipate that a key marker of trustworthiness will be interfaces that do not mislead. 
    
    Interacting with the explicit information displayed should not cause users to implicitly draw incorrect assessments about the model (we see a similar principle articulated by Mackinlay for data visualization ).

    Undoubtedly, the interfaces we present in this article have room to improve in this regard. 

    Fundamental research, at the intersection of machine learning and human-computer interaction, is necessary to resolve these issues.

Trusting our interfaces is essential for many of the ways we want to use interpretability.

    This is both because the stakes can be high (as in safety and fairness) and also because ideas like training models with interpretability feedback put our interpretability techniques in the middle of an adversarial setting.

## Conclusion & Future Work

There is a rich design space for interacting with enumerative algorithms, and we believe an equally rich space exists for interacting with neural networks.

    We have a lot of work left ahead of us to build powerful and trustworthy interfaces for interpretability.

    But, if we succeed, interpretability promises to be a powerful tool in enabling meaningful human oversight and in building fair, safe, and aligned AI systems.
