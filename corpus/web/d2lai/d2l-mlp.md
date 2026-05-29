---
title: "Dive into Deep Learning 1.0.3 documentation"
source: "web"
url: "https://d2l.ai/chapter_multilayer-perceptrons/index.html"
domain: "d2l.ai"
name: "d2l - MLP"
fetched_at: "2026-05-26T23:30:15Z"
topics: ["mlp"]
aliases: ["Dive into Deep Learning 1.0.3 documentation"]
tags: [topic/neural-network-foundations, level/intermediate, medium/article, task/general, technique/mlp]
---

Title 

![Dive into Deep Learning](../_static/logo-with-text.png)

Table Of Contents

 Title 

![Dive into Deep Learning](../_static/logo-with-text.png)

Table Of Contents

# 5. Multilayer Perceptrons Â¶

In this chapter, we will introduce your first truly deep network. The
simplest deep networks are called multilayer perceptrons , and they
consist of multiple layers of neurons each fully connected to those in
the layer below (from which they receive input) and those above (which
they, in turn, influence). Although automatic differentiation
significantly simplifies the implementation of deep learning algorithms,
we will dive deep into how these gradients are calculated in deep
networks. Then we will be ready to discuss issues relating to numerical
stability and parameter initialization that are key to successfully
training deep networks. When we train such high-capacity models we run
the risk of overfitting. Thus, we will revisit regularization and
generalization for deep networks. Throughout, we aim to give you a firm
grasp not just of the concepts but also of the practice of using deep
networks. At the end of this chapter, we apply what we have introduced
so far to a real case: house price prediction. We punt matters relating
to the computational performance, scalability, and efficiency of our
models to subsequent chapters.

- 5.1. Multilayer Perceptrons 5.1.1. Hidden Layers 5.1.2. Activation Functions 5.1.3. Summary and Discussion 5.1.4. Exercises

- 5.2. Implementation of Multilayer Perceptrons 5.2.1. Implementation from Scratch 5.2.2. Concise Implementation 5.2.3. Summary 5.2.4. Exercises

- 5.3. Forward Propagation, Backward Propagation, and Computational Graphs 5.3.1. Forward Propagation 5.3.2. Computational Graph of Forward Propagation 5.3.3. Backpropagation 5.3.4. Training Neural Networks 5.3.5. Summary 5.3.6. Exercises

- 5.4. Numerical Stability and Initialization 5.4.1. Vanishing and Exploding Gradients 5.4.2. Parameter Initialization 5.4.3. Summary 5.4.4. Exercises

- 5.5. Generalization in Deep Learning 5.5.1. Revisiting Overfitting and Regularization 5.5.2. Inspiration from Nonparametrics 5.5.3. Early Stopping 5.5.4. Classical Regularization Methods for Deep Networks 5.5.5. Summary 5.5.6. Exercises

- 5.6. Dropout 5.6.1. Dropout in Practice 5.6.2. Implementation from Scratch 5.6.3. Concise Implementation 5.6.4. Summary 5.6.5. Exercises

- 5.7. Predicting House Prices on Kaggle 5.7.1. Downloading Data 5.7.2. Kaggle 5.7.3. Accessing and Reading the Dataset 5.7.4. Data Preprocessing 5.7.5. Error Measure 5.7.6. \(K\) -Fold Cross-Validation 5.7.7. Model Selection 5.7.8. Submitting Predictions on Kaggle 5.7.9. Summary and Discussion 5.7.10. Exercises

Previous
4.7. Environment and Distribution Shift

Next
5.1. Multilayer Perceptrons
