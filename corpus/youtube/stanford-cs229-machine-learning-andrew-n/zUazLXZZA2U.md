---
title: "Lecture 11 - Backprop & Improving Neural Networks | Stanford CS229: Machine Learning (Autumn 2018)"
source: "youtube"
video_id: "zUazLXZZA2U"
url: "https://www.youtube.com/watch?v=zUazLXZZA2U"
channel: "Stanford CS229 - Machine Learning (Andrew Ng, 2018)"
uploader: "Stanford CS229 - Machine Learning (Andrew Ng, 2018)"
duration_sec: 4598
language: "en-en"
fetched_at: "2026-05-26T22:52:08Z"
topics: []
---

[0:03] Hi everyone. [NOISE] Welcome, welcome to the second lecture on deep learning for CS229. So a quick announcement before we start. There is a Piazza post Number 695 which is the mid-quarter survey for CS229, so fill it in when you have time. Okay. So let's get back to deep learning. So last week together we've seen, uh, what a neural network is and we started by

[0:35] defining the logistic regression from a neural network perspective. We said that logistic regression can be viewed as a one-neuron neural network where there is a linear part and an activation part which was sigmoid in that case. We se- we've seen that sigmoid is a common activation function to be used for classification tasks because it casts a number between minus infinity and plus infinity in 0, into 0, 1 interval which can be interpreted as a probability. And then we introduced the neural network,

[1:06] so we started to stack some neurons inside a layer and then stack layers on top of each other and we said that the more we stack layers the more parameters we have, and the more parameters we have, the more our network is able to copy the complexity of our data because it becomes more flexible. So, uh, we stopped at a point where we did a forward propagation, we had an example during training, we forward propagated through the network, we get the output, then we compute the cost function which compares this output to the ground truth,

[1:39] and we were in the process of backpropagating the error to tell our parameters how they should move in order to detect cats more properly. Does that make sense for this part? So today, we're going to continue that. So we're in the second part, neural networks, we're going to derive the backpropagation with the chain rule and after that, ah, we're going to talk about how to improve our neural networks. Because in practice, it's not because you designed a neural network that it's going to work, there's a lot of hacks and tricks that you

[2:10] need to know in order to make a neural network work. Okay, let's go. So first thing that we talked about is in order to define our optimization problem and find the right parameters, we need to define a cost function, and usually we said we would use the letter j to denote the cost function. So here, when I talk about cost function, I'm talking about the batch of examples. It means I'm forward propagating m examples at a time.

[2:41] You remember why we do that? What's the reason we use a batch instead of a single example? Vectorization. We want to use what our GPU can do and parallelize the computation. So that's what we do. So we have m examples that go- forward propagate in the network. And each of them has a loss function associated with them, the average of the loss functions over the batch give us the cost function. And we had defined these loss function together.

[3:15] L of i. Assuming we're still, and just as a reminder, we're still in this network where, where we had a cat, remember? This one. Remember this guy. x_1 to x_n. The cat was flattened into a vector, RGB matrix into one vector and then there was a neural network with three neurons, then two neurons, then one neuron. Remember? Fully-connected here. Everything. Up, up,

[3:51] and then we add y hat. You remember this one? I think that was this one here. Yeah, okay. So now, we're here, we take m images of cats or non-cats, forward propagate everything in the network, compute our loss function for each of them, average it, and get the cost function. So our last function was the binary cross-entropy or also called the loss function- the logistic loss function and it was the following. y_i log of y hat i plus 1 minus y_i log of 1 minus y hat i.

[4:30] So let me circle this one, it's an important one. And what we said is that this network has many parameters. And we said, the first layer has w_1, b_1, the second layer has w_2, b_2, and the third layer has w_3, b_3 where the square brackets dis- denotes the layer. And we have to train all these parameters.

[5:02] One thing we notice is that because we want to make a good use of the chain rule, we're going to start by, by computing the derivative of these guys, w_3 and b_3 and then come back and do w_2 and b_2 and then back again w_1 and b_1. In order to use our formulas of the update of the gradient descent where w would be equal to w minus Alpha derivative of the cost with respect to w and this for any layer l between 1 and 3, same for b.

[5:41] Okay, so let's try to do it. This is the first number we want to compute. And remember, the reason we want to compute derivative of the cost with respect to w_3 is because the relationship between w_3 and the cost is easier than the relationship between w_1 and the cost because w_1 had much more connection going through the network before ending up in the cost computation. So one thing we should notice

[6:15] before starting this calculation is that the derivative is linear. So this, if I take the derivative of j, I can just take the derivative of l, and it's the same thing, I just need to add the summation prior to that because derivative is a linear operation. That makes sense to everyone? So instead of computing this, I'm going to compute that and then I will add the summation, it will just make our notation easier. So I'm taking the derivative of a loss of

[6:47] one example propagated to the network with respect to w_3. So let's do the calculation together. I have a 1, I have a minus y_i derivative with respect to w_3, of what? We remember that y hat was equal to sigmoid of w_3 x plus b or w_3 a_2 plus b because a_2 is the input to the second layer, remember. So I would write it down here, sigmoid of w_3 a_2 plus b_3.

[7:23] Okay? Yeah. It's good like that? It's too small? w_3 a_2 plus b_3.

[7:54] It's good like that, yeah? Okay. So we have this term and then we have the second term which is plus 1 minus y_i times derivative of w_3. Derivative with respect to w_3 of 1. Oh sorry, I forgot the logarithm here. Of log of 1 minus sigmoid of w_3 a_2 plus b_3.

[8:30] And so just a reminder, the reason we have this is because we've written the forward propagation in the previous class. You guys remember the pro- forward propagation? We had z_3, which took a_2 as inputs and computed the linear part, as sigmoid is- is the activation function used in the last neuron over here. Okay. So let's try to- to compute this derivative. y_i, so the derivative of log, [NOISE] log prime equals 1 over log.

[9:03] Remember this- this- this formula, so I will just take 1 over, sorry, 1 over x minus- 1  over x if you put an x here. So log prime of x. So I will take one over sigmoid of w_3 a_2 plus b_3. I know that thing can be written a_3, right? So I will just write a_3 instead of writing the single a again. So we have 1 over a_3 times the derivative of a_3 with respect to w_3.

[9:36] We remember that, I'm going to write it down here. If we take the derivative of sigmoid of blah, blah, blah. Let's say, derivative of log of sigmoid over w. What we have is 1 over the sigmoid times the derivative with respect to w_3 of the sigmoid. Does that makes sense? That's what we're using here. So the derivative of sigmoid, sigmoid-prime of x is actually pretty easy to compute.

[10:09] It's sigmoid of x times 1 minus sigmoid of x. Okay. So I'm just going to take the derivative. It's going to give me a- a_3 times 1 minus a_3. There's still one step because there is a composition of three functions here. There is a logarithm, there's a sigmoid, and there is also a linear function, w_x plus b or w a_2 plus b. So I also need to take the derivative of the linear part with respect to w_3.

[10:43] Because I know that sigmoid of w_3, a_2 plus b_3. If I wanna take the derivative of that with respect to w_3, I need to go inside and take the derivative of what's inside, okay? So this will give me the sigmoid or whatever a_3 times 1 minus a_3 times the derivative with respect to w_3 of the linear part.

[11:15] [NOISE] Does this make sense? So I am going to write it here bigger. Here, I need to take the derivative of the linear part with respect to w_3, which is equal to a_2 transpose. So one thing you- you may wanna check, is when we compute- when I'm trying to compute this derivative.

[11:51] [NOISE] I'm trying to compute this derivative. Why is there a transpose that comes out? How do you come up with that? You look at the shape here. What's the shape of w_3? Someone remembers?

[12:23] 1 by 2. Yeah, why 1 by 2? [BACKGROUND] Yeah, it's connecting two neurons to one neuron. So it has to be 1 by 2. Usually flip it. And in order to come back to that, you can write your forward propagation, make the shape analysis, and find out that it's a 1 by 2 matrix. How about this thing? What's the shape of that?

[12:53] [NOISE]. The scalar. It's a scalar, yeah. So scalar. So it's 1 by 1. How do you know? It's because this thing is basically z_3. It's the linear part of the last neuron and a_3, we know that it's y-hat. So it's a scalar between 0 and 1. So this has to be a scalar as well. Because taking the sigmoid should not change the shape. So now, the question is what's the shape of this entire thing?

[13:24] The shape of this entire thing should be the shape of w_3 because you're taking the derivative of a scalar with respect to a higher-dimensional matrix or vector here called a row vector. Then it means, that the shape of this has to be the same shape of w_3. So 1 by 2. And you know that when you take this simple derivative in- in real life, like in- in, uh, with scalars, not with high-dimensional, you know that this is an easy derivative.

[13:54] It just should- it should give you a_2, right? But in higher dimension, sometimes you have transpose that come up. And how do you know that the answer is a_2 transpose? It's because you know that a_2 is a 2 by 1 matrix. [NOISE] So this is not possible. It's not possible to get a_2, because otherwise it wouldn't match the derivative that you are calculating. So it has to be a_2 transpose. So either you- you learn the formula by heart or you- you learn how to analyze shapes, okay? Any questions on that?

[14:28] Okay. So that's why it's a_2 transpose. Now, l minus y_i. So I'm- I'm on this one now. The second term of the- of the derivative. And I take the derivative of this. So I get 1 over 1 minus a_3. a_3 denotes the sigmoid. So I'm just copying this back using the fact that the derivative of the logarithm is 1 over x,

[14:58] and then I will multiply this by the derivative of 1 minus a_3 with respect to w_3. I know that there is a minus that needs to come up. So I will write it down here, minus 1 and I also have the derivative of the sigmoid with respect to what's inside the sigmoid. So a_3 [NOISE] times 1 minus a_3. And what's the last term? The last term is simply the one we just talked about. It's the derivative of what's inside the sigmoid with respect to w_3.

[15:30] So it's a_2 transpose again. Okay. So now, I will just simplify. I know this scalar simplifies with this one. This one simplifies with that one. We're going to copy back all the results minus [NOISE] y_i times 1 minus a_3

[16:03] a_2 transpose plus 1 minus y_i times the minus- I'm going to put the minus here. So I'm taking the minus putting it on- on the front times a_3 times a_2 transpose. And then, quickly looking at that I see that some of the terms will cancel out, right?

[16:35] Okay. So I have one term here, y-hat- y_i times minus a_3 a_2 transpose would cancel out with plus y_i a_3 a_2 transpose. This makes sense? So like, the term that we multiply this number, we cancel out with the term, we multiply this number. We need to continue. [NOISE] It gives me y_i times a_2 transpose, this part,

[17:07] minus a_3 times a_2 transpose. I, I can factor this because I have the same term a_2 transpose. And it gives me finally, y_i minus a_3 times a_2 transpose. Okay, so it doesn't look that bad actually. I don't know, when- when we take a derivative of something kin- kinda ugly we- we expect something ugly to come out but this doesn't seem too bad.

[17:44] Any questions on that? I let you write it quickly, and then we're going to move through to the rest. So once I get these results, I can just write down the costs of the derivative with respect to w_3. I know it's just one minus. I just need to- to take the summation of this thing. So y_i minus a_3 times [NOISE] y_2 transpose- a_2 transpose.

[18:15] And I have a minus sign coming upfront. So that's my derivative. [NOISE] Okay. So we're done with that. And we can, we can just take this formula, plugging it back in our gradient descent update rule, and update w_3. Yeah. Now, the question is, you can do the same thing as, as we just did but with b_3.

[18:47] It's going to be the similar difficulty. We're going to do it with w_2 now, and think how does that backpropagate to w_2. So now it's w_2 star. We want to compute the derivative of l, the loss, with respect to w of the second layer. The question is how I'm gonna get this one without having too much work. I'm not gonna start over here as we said last time, I'm going to use the chain rule of calculus. So I'm going to try to decompose this derivative into several derivatives.

[19:21] So I know that y hat is the first thing that is connected to the loss function, right. The output neuron is directly connected to the loss function. So I'm going to take the derivative of the loss function with respect to y hat, also called a_3. Right? This is the easiest one I can calculate. I also know that a_3, which is the output activation of the last neuron, is connected with the linear part of the last neuron, which is z_3. So I can take the derivative of a_3 with respect to z_3.

[19:54] Do you remember what this is going to be? Derivative of a_3 with respect to z_3? Derivative of Sigmoid. I know that a_3 equals Sigmoid of z_3. So this derivative is very simple. It's just that. It's just a_3 times 1 minus a_3. All right. So I'm going to continue. I know that z_3, z_3 is equal to what? It's equal to w_3, a_2 plus b.

[20:24] Which path did I need- do I need to take in order to backpropagate? I don't wanna take the derivative with respect to w_3 because I will only get stuck. I don't wanna take the derivative with respect to b_3 because I will get stuck. I will take the derivative with respect to a_2. Because a_2 will be connected to z_2, z_2 will be connected to a_1, and I can backpropagate from this path. So I'm going to take derivative of z_3 with respect to a_2 to have my error backpropagate, and so on. I know that a_2 is equal to Sigmoid of z_2.

[20:59] So I'm just going to do that. And I know that this derivative is going to be easy as well. And finally, I also know that z_2 is connected to w_2. So I'm going to take derivative of z_2 with respect to w_2. So just what I want you to get is the thought process of this chain rule. Why don't we take a derivative with respect to w_3 or b_3? It's because we will get stuck. We want the error to back propagate. And in order for the error to backpropagate, we have to go through variables that are connected to each other. Does this makes sense?

[21:37] So now the question is how can we use this? How can we use the derivative we already have in order to, to, to, to compute the derivative with respect to w_2? Can someone tell me how we can use the results from this calculation, in order not to do it again? Cache it. You cache it? Um, so there's another discussion on caching,

[22:11] which is, which is correct that in order to get this result very quickly we will use cache. But, uh, what I want here is to- you to tell me if these results appear somewhere here. Yeah? [inaudible] the first three terms. The first three terms. So this one, this one, and this one? I'm not sure. Yeah. Is it the first two terms or the first three terms? Two. The first two terms. Yeah. But good intuition. Yeah. So this result is actually the first two terms here. We just calculated it.

[22:42] Okay. What- how do we know that? It's not easy to see. One thing we know based on what we've written very big on this board is that the derivative of z_3, because this is z_3, right? Derivative of z_3 with respect to w_3 is a_2 transpose. Right. So I could write here that this thing is derivative of z_3 with respect to w_3. Is it correct? So I know

[23:14] that because I wanted to compute the derivative of the loss to w_3, I know that I could have written derivative of loss with respect to w_3 as derivative of loss with respect to z_3, times derivative of z_3 with respect to w_3. Correct. And I know that this is a_2 transpose. So it means that this thing is the derivative of the loss with respect to z_3. Does that make sense? So I got,

[23:48] I got my decomposition of the derivative we had. If we wanted to use the chain rule from here on, we could have just separated it into two terms, and took the derivative here. Okay. So I know the result of this thing. I know that this thing is basically a_3 minus y, times a_2 transpose. I just flipped it because of the minus sign. Okay. Is it mine? [NOISE].

[24:20] Okay. [NOISE]. Now, tell me what's this term. What is this term? Let's go there. Yeah. Sigmoid. So Sigmoid. I'm just going to write it a_2 times 1 minus a_2. Does that make sense? Sigmoid times 1 minus Sigmoid. What is this term? Uh, oh sorry my bad.

[24:58] That's not the right one. This one, this one is that. This one is Sigmoid. a_2 is Sigmoid of z_2. So this result comes from this term. Was- what about this term? w_3. Sorry. w_3. Is it w_3 or no? I heard transpose. How do we know if it's w_3 or w_3 transpose? So let's look at the shape of this. What's z_3?

[25:30] One by one. It's one by one. It's a scalar. It's the linear part of the last neuron. What's the shape of that? This is 2, 1. We have two neurons in the layer. w_3. We said that it was a 1 by 2 matrix, so we have to transpose it. So the result of that is w_3 transpose. And how about the last term?

[26:01] Same as here. One layer before. Yeah, someone said they won't transpose. Okay. Yeah? The numbers are [inaudible] that one. This one? Yeah. There is a transpose here. [inaudible] w_5.

[26:39] Oh yeah, yeah. You're correct. You're correct. Thank you. That's what you mean? Yeah. Yeah. This one was from the z_3, to w_2. We didn't end up using that because we will get stuck, so there's no a_2 transpose here. Thanks. Any other questions or remarks? So that's cool. Let's, let's, let's write- let's write down our derivative cleanly on the board. So we have derivative of our loss function with respect to w_2,

[27:16] which seems to be equal to a_3 minus y, from the first term. The second term seems to be equal to, uh, w_3 transpose. Then we have a term which is a_2 times 1 minus a_2. Okay. And finally, finally we have another term that is a_1 transpose.

[27:52] So are we done or not? So actually there is that- the thing is there's two ways to compute derivatives. Either you go very rigorously and do what we did here for w_2, or you try to do a chain rule analysis, and you try to fit the terms. The problem is this result is not completely correct.

[28:22] There is a shape problem. It means when we took our derivatives, we should have flipped some of the terms. We didn't. There is actually- we, we won't have time to go into details in this lecture because we have other things to see, but there is, uh, a section note I think on the website, which details the other method which is more rigorous, which is like that for all the derivatives. What we are going to see is how you can use chain rule plus shape analysis to come up with the results very quickly. Okay. So let's, let's analyze the shape of all that. We know that the first term is a scalar.

[28:53] It is a 1 by 1. We know that the second term is the transpose of 1 by 2. So it's 2 by 1. And we know that this thing here a_2 times 1 minus a_2 is, uh, 2 by 1. It's an element-wise product. And this one is a_1 transpose, so it's 3 by 1 transpose. So it is 1 by 3. So there seems to be a problem here. There is no match between these two operations for example.

[29:24] Right? So the question is, how- how can, we how can we put everything together? If we do it very rigorously, we know how to put it together. If you're used to doing the chain rule, you can quickly sh- quickly do it around. So after experience, you will be able to, to fit all these together. The important thing to know is that here there is an element-wise product, which is here. So every time you will take the derivative of the Sigmoid it's going to end up being an element-wise product.

[29:56] And it's the case whatever the activation that you're using is. So the right result is this one. So here I have my element-wise product of a 2 by 1 [NOISE] by a 2 by 1. So it gives me a 2 by 1 column vector and then I need something that is 1 by 1 and 1 by 3.

[30:28] How do I know, wha- what I need to have, I know that the shape of this thing. W3 needs to be 2 by 3. It's connecting three neurons to two neurons. So W2 has to be 2 by 3. In order to end up with this, I know that this has to come here A3 minus y and A1 transpose comes at the end. And here I get my correct answer.

[31:10] Don't worry if it's the first time th- the chain rule is going quickly, don't worry. Read the lecture notes with the rigorous parts. Taking the derivative, it will make more sense. But I feel it's, uh, usually in practice, we don't compute these chain rules anymore, uh, because- because programming frameworks do it for us but it's important to know at least how the chain rule decomposes, uh, and also how to make these, compute these derivatives. If you read research papers specifically.

[31:41] Any questions on that? I think I wanna go back to what you mentioned with the cache. So why is cache very important? That was your question as well? [BACKGROUND] Yeah, yeah it has to be. Right. So it means when you take the derivative of Sigmoid, you take derivative with respect to every entry of the matrix which gives you an element-wise product. Um, going back to the cache. So one thing is, it seems that during backpropagation,

[32:13] there is a lot of terms that appear that were computed during forward propagation. Right. All these terms; a1 transpose, a2, a3, all these, we have it from the forward propagation. So if we don't cache anything, we have to recompute them. It means I'm going backwards but then I feel, oh, I need a2 actually. So I have to re- go forward again to get a2. I go backwards, I need a1. I need to forward propagate my x again to get a1. I don't wanna do that. So in order to avoid that, when I do my forward propagation,

[32:45] I would keep in memory almost all the values that I'm getting including the Ws because as you see to compute the derivative of loss with respect to W2 we need W3, but also, the activation or linear variables. So I'm going to save them in my, in my network during the forward propagation in order to use it during the backward propagation. So it makes sense. And again, it's all for computational ef- efficiency.

[33:15] It has some memory costs. Okay. So that was backpropagation. And now I can use my formula of the costs with respect to the loss function. And I know that this is going to be my update. [NOISE] This is going to be used in order to update W2 and I will do the same for W1.

[33:49] Then you guys can do it at home. If you wanna meet, wanna make sure you understood, take the derivative with respect to W1. Okay. So let's move on to the next part, [NOISE] which is improving your neural network. So in practice, when you, when you do this process of training forward propagation,

[34:20] backward propagation updates, you don't end up having a good network mo- most of the time. In order to get a good network, you need to improve it. You need to use a bunch of techniques that will make your network work in practice. The first, the first trick is to use different activation functions. So together, we've seen one activation function which was Sigmoid.

[34:52] And we remember the graph of Sigmoid is getting a number between minus infinity and plus infinity and casting it between 0 and 1. And we know that the formula is Sigmoid of z equals 1 over 1 plus exponent so minus z. We also know that the derivative of Sigmoid is Sigmoid of z times 1 minus Sigmoid of z. Okay. Another very common, uh, activation function is ReLU.

[35:24] We talked quickly about it last time. ReLU of z which is equal to 0 if z is less than 0 and z if z is positive. So the graph of ReLU looks like something like this. And finally, another one we were using commonly as well is

[35:55] tan h. So hyperbolic tangents and tan h of z exponential z minus exponential minus z over exponential z plus exponential minus z. The derivative of tan h is 1 minus tan h squared of z. And the graph looks kind of like Sigmoid,

[36:29] but, but it goes between minus 1 and plus 1. So one question. Now that I've given you three activation functions, can you guess why we would use one instead of the other and, and which one has more benefits? So when I talk about activation functions, I talk about the functions that you will put in these neurons after the linear part.

[37:08] What do you think is the main advantage of Sigmoid? Yeah. We use it for classification. Yep. You use it for classification, between it gives you a probability. What's the main disadvantage of Sigmoid? It's easy. That should be an advantage, should be a benefit. Yeah? [BACKGROUND]

[37:43] Correct. If you're at high activation, if you are at high z's or low z's, your gradient is very close to 0. So look here. Based on this graph we know that if z is very big. If z is very big our gradient is going to be very small, the slope of this, of this graph is very, very small. It's almost flat. Same for z's that are very low in the negative. Right. What's the problem with having low gradients is when I'm back propagating. If the z I cached was big, the gradient is going to be very small and it will be super hard to update

[38:17] my parameters that are early in the network because the gradient is just going to vanish. Does that makes sense? So Sigmoid is one of these activations which, which works very well in the linear regime, but has trouble working in saturating regimes because the network doesn't update the parameters properly. It goes very, very slowly. We're going to talk about that a little more. How about tan h? Very similar, right? Similar like high z's and low z's lead to saturation of a tan h activation.

[38:52] ReLU on the other hand doesn't have this problem. If z is very big in the positives, there is no saturation. The gradient just passes and the gradient is 1, when we were here. The slope is equal to 1. So it's actually just directing the gradient to some entry. Is not multiplying it by anything when you backpropagate. So you know this term here, this term that I have here. All the a3 minus a3 times 1 minus a3 or 1 minus a2. If we use ReLU activations,

[39:24] we would change this with what's- with- with the derivative of ReLU and the derivative of ReLU can be written indicator function of z being positive. You've seen indicator functions. So this is equal to 1 if z is positive, 0 otherwise. Okay. So we will see why we use ReLU mostly. Yeah? [BACKGROUND]

[40:00] Yeah. You remember the house prediction example? In that case, if you want to, if you want to predict the price of a house based on some features, you would use ReLU. Because you know that the output should be a positive number between 0 and plus infinity, it doesn't make sense to use 1 of tan h or similar. Yep. [BACKGROUND] Doesn't really matter. I think if, if I want my output to be between 0 and 1 I would use Sigmoid, if I want my output to be between minus 1 and

[40:31] 1 I would use tan h. So you know, there is, there are some tasks where the output is kind of a reward or a minus reward that you want to get. Like in reinforcement learning, you would use tan h as an output activation which is because minus 1 looks like a negative reward, plus 1 looks like a positive reward, and you want to decide what should be the reward. Why do we consider these functions? Good question. Why do we consider these functions? We can actually consider any functions apart

[41:03] from the identity function. So let's see why. Thanks for the transition. [LAUGHTER] Like why do we need activation functions? So let's assume that we have a network which is the same as before. So our network is three neurons casting into two neurons casting into one neuron, ah, and we're trying to use activations are equal to identity functions.

[41:42] So it means z is given to z. Let's try to derive the forward propagation, y_hat equals a_3, equals z_3, equals w_3, a_2 plus b_3. I know that a_2, a_2 is equal to z_2 because there is no activation and z_2 is equal to w_2 a_1 plus b_2.

[42:14] So I can cast here w_2, w_2 a_1 plus b_2 plus b_3. I can continue. I know that a_1 is equal to z_1, and I know that z_1 is w_1 x plus b,

[43:25] and b equals w_3 times w_2 times b_1 plus w_3 times b_2 plus b_3. So what's the insight here?

[43:56] Is that we need activation functions. The reason is, if you don't choose activation functions, no matter how deep is your network, it's going to be equivalent to a linear regression. So the complexity of the network comes from the activation function. And the reason we can understand- if we're trying to detect cats, what we're trying to do is to train a network that will mimic the formula of detecting cats. We don't know this formula, so we want to mimic it using a lot of parameters. If we just have a linear regression,

[44:28] we cannot mimic this because we are going to look at pixel by pixel and assign every weight to a certain pixel. If I give you an example, it's not going to work anymore. Yeah, yeah. So I think that's, that, that goes back to your question as well. So this is why we need activation functions. And then the question was, can we use different activation functions and how do we, how do we put them inside a layer or inside neurons? Usually, we would use,

[44:59] there are more activation functions. I think in CS230 we'll go over a few more but not, not, not today. These have been designed with experience, so these are the ones that's, that, that's work better and lets our networks train. There are plenty of other activation functions that have been tested. Usually, you would, you would, uh, use the same activation functions inside every layer. So when you, it's, it's a, it's, it's for, for training. It doesn't have any special reason I think but when you have a network like that,

[45:33] you would call this layer a ReLU layer meaning it's a fully connected layer with ReLU activation. This one a Sigmoid layer, it means it's a fully connected layer with the Sigmoid activation. And the last one is Sigmoid. I, I think people have been trying a lot of putting, activat- different activations in different neurons in a layer, in different layers and the consensus was using one activation in the layer and also using one of these three activations. Yeah. So if someone comes up with a better activation that is

[46:07] obviously helping training our models on different datasets, people would adopt it but right now these are the ones that work better. And you know, last time we talked about hyper-parameters a little bit. These are all hyper-parameters. So in practice, you're not going to choose these randomly, you're going to try a bunch of them and choose some of them that seem to help your model train.

[46:38] There's a lot of experimental results in deep learning and we don't really understand fully why certain activations work better than others. Okay, let's move on. [NOISE]

[47:14] Okay, let's go over initialization techniques. [NOISE] Uh, actually, let me use this board. So another trick that you can use

[47:51] in order to help your network train are initialization methods and normalization methods. So, um, earlier we talked about the fact that if z is too big, or z is too low in the negative numbers, it will lead to saturation of the network. So in order to avoid that you can use normalization of the input.

[48:26] So assume that you have a network where the data is two-dimensional, x_1, x_2 is our two-dimensional input. You can assume that x_1, x_2 is distributed like this, let's say. So this is if I plot x_1 against x_2 for a lot of data, I will get that type of graph. Uh, the problem is that if I do my wx plus b, to compute my z_1,

[48:57] if xs are very big, it will lead to very big zs which will lead to saturated activations. In order to avoid that, one method is to compute the mean of this data using Mu equals 1 over the size of the batch of data that you have in the training sets. Sum of xis. So it's just giving you the mean for x_1, and the mean for x_2.

[49:27] You would compute the operation x equals x minus Mu, and you will get that type of plot. If you replot the transform data, let's say x_1 tilde, x_2 tilde. So here is a little better, but it's still not good. In order to solve the problem fully, we are going to compute Sigma squared, which is basically the standard deviation squared, so the variance of the data,

[50:00] and then you will divide by, uh, Sigma squared. So you would do that and you would make the transformation of x being equal to x divided by Sigma, and it will give you a graph that is centered up here.

[50:37] So you, you usually prefer to, to work with a centered data. Yeah? [inaudible] tilde? Sorry, oh yeah, yeah, sorry, sorry, yeah, correct. So if we subtract the mean of x_1 and x_2, it will be [inaudible]. Sorry, it should look like this, but it would be centered.

[51:13] Okay, and then if you stan- if you standardize it, it looks like something like that. So why is it better? Because if you look at you- your loss function now, before the loss function would look like something like this. [NOISE] And after normalizing the inputs, it may look like something, something like this. So what's the difference between these two loss functions?

[51:44] Why is this one easier to train? It's because if you have the starting point that is here let's say, their gradient descent algorithm is going to go to towards approximately the steepest slope. So we're going to go like there, and then this one is going to go there, and then you're going to go there, and then you're going to go there like that and so on, until you end up at the right points. But the steeper slope in this loss contour is always pointing towards the middle. So if you start somewhere,

[52:15] it will directly go towards the minimum of your loss function. So that's why it's helpful usually to normalize. So this is one method, uh, and in practice, the way you initialize your weights is very important. Yeah? [BACKGROUND] Uh, yes. So. [BACKGROUND]

[52:47] Exactly. So here I used a very simple case but you would divide elementwise by, by the Sigma here, okay? So like every entry of your matrix you would divide it by the Sigma. One, one other thing that is important to notice. This Sigma and Mu are computed over the training set. You have a training set, you compute the mean of the training, set the standard deviation, of the training set, and these Sigma and Mu have to be used on the test set as well. It means now that you want to test your algorithm on the test set,

[53:17] you should not compute the mean of the test set, and the standard deviation of the test set and normalize your test inputs through the network. Instead, you should use the Mu and the Sigma that were computed on the train set because your network is used to seeing this type of transformation as an input. So you want the distribution of the inputs at the first neuron to be always the same, no matter if it's a train or the test set. What you do is that [inaudible]

[53:49] Here? Likely, yeah. This leads to fewer iterations. Okay, we have a lot to see so I will, I will skip a few questions. So let's, let's delve a little more into vanishing and exploding gradients. So in order to get an intuition of why we

[54:22] have these vanishing or exploding gradient problem, we can consider a network which is very, very deep and has a two-dimensional input, okay? And so on. So let's say we have, let's say we have ten layers in total. Ten layers plus an output layer. So assume, assume all the activations are identity functions,

[54:58] and assume that these biases are equal to 0. If you compute y hats, the output of the network with respect to the input. You know that y hat will be equal to w of layer L, capital L denotes the last layer, times a l minus 1 plus bL, but bL is 0 so we can remove it. w_l times a_L minus 1. You know that a_L minus 1 is w_l minus 1

[55:33] times a_L minus 2 because the activation is an identity function and so on. You can back propagate, you can go back and you will get that y hat equals w_L times w_l minus 1 times blah, blah, blah, times w_1 times x. You get something like that, right? So now, let's consider two cases.

[56:05] Let us consider the case where the w_l matrices are a little bigger than the identity function, a little larger than the identity function in terms of values. Let's say w_l, including all these. So all these matrices which are 2 by 2 matrices, right, are these ones. What's the consequence? The consequences that this whole thing here is going to be equal to 1.5 to the power L,

[56:42] 1.5 to the power L, 0, 0. It will make y hat explode. It will make the value of y hat explode, just because this number is a tiny little bit more than 1. Same phenomenon, if we had 0.5 instead of 1.5 here, the value, the multiplicative value of all these matrices will be 0.5 to the power L here, 0.5 to the power L here, and y hat will always be very close to 0.

[57:16] So you see, the issue with vanishing exploding gradients is that all the errors add up like multiply each other. And if you end up with numbers that are smaller than one, you will get a totally vanished gradient. When you go back, if you have values that are a little bigger than 1 you will get exploding gradients. So we did it as a forward propagation equation, we could have done it exactly the same analysis. We did derivatives, assuming the derivatives of the weight matrices are a little lower than the identity,

[57:50] or a little higher than the identity. So we want to avoid that. One way that is not perfect to, to avoid this is to initialize your weights properly, initialize them into the right range of values. So you agree that we would prefer the weights to be around 1, as close as possible to 1. If they're very close to 1, we probably can avoid the vanishing and exploding gradient problem. So let's look at the initialization problem.

[58:24] The first thing to look at is example of the one neuron. [NOISE] If you consider this neuron here, which has a bunch of inputs and outputs and activation a. [NOISE] You know that the equation inside the neuron is a equals whatever function, let's say sigmoid of Z and you know

[58:58] that z is equal to W_1 X_1 plus W_2 X_2 plus blah, blah, blah plus W_n X_n. So it is a dot product between the W's and the X's. So the interesting thing to notice is that we have n terms here. So in order for Z to not explode, we would like all of these terms to be small. If W's are too big, then this term will explode with the size of the inputs of the layer.

[59:31] So instead if we have a large n, it means the input is very large, what we want is very small W_i's. So the larger n, the smaller it has to be W_i. So based on this intuition, it seems that it would be a good idea to initialize W_i's with something that is close to 1 over n. We have n terms, the more terms we have, the more likely Z is going to be big.

[1:00:03] But if our initialization says the more terms you have, the smaller the value of the weights, we should be able to keep Z in a certain range that is appropriate to avoid vanishing and exploding gradients. So this seems to be a possible initialization scheme. So in practice, I'm going to write a few initialization schemes that we're not gonna prove. If you're interested in seeing more proofs of that, you can take CS230, where we prove this initialization scheme.

[1:00:41] May I take down the board? So there are a few initializations that are commonly used and again, this is, this is very practical and people have been testing a lot of initializations, but they ended up using those. [NOISE] So one is to initialize the weights. I'm writing the code for those of you who know numPy. I'm not gonna compile it here. With whatever shape you are using,

[1:01:14] elementwise times the square root of 1 over n of L minus 1. So what does that mean? It means that I will look at the number of inputs. I'm writing an L minus 1 here, n to the L minus 1. I'm looking at how many inputs are coming to my layer assuming we're at layer L. How many inputs are coming. I'm going to initialize the weights of

[1:01:46] this layer proportionally to the number of inputs that are coming in. So the intuition is very similar to what we described there. So this initialization has been shown to work very well for sigmoid activations. So if you use sigmoid. What's interesting is if you use ReLU, it's been, it's been observed that putting a 2 here instead of a 1 would make the network train better.

[1:02:17] And again, it's very practical. It's one of the fields that, that we need more theory on it, but a lot of observations had been made so far. Do you guys want to just do that as a project to see why is this happening? It would be interesting. Okay. [NOISE] And finally, there is a more common one that is used which is called the Xavier initialization,

[1:02:54] which proposes to update the weights [NOISE] using, uh, square root of 1 over n_ l minus 1 for tan h. This is another one. And another one that is I believe called Glorot initialization recommends to initialize the weights of a layer using the following formula.

[1:03:31] So quickly, the, the quick int- intuition behind the last one. The last one is, is very often used. The quick intuition is that we're doing the same thing but also for the backpropagated gradients. So we're saying the weights are going to multiply the backpropagated gradients. So we also need to look at, at how many inputs do we have during the backpropagation. And L is the number of inputs you have during backpropagation and L minus 1 is the number of inputs you have during forward propagation. So taking an average, a geometric average of those.

[1:04:02] [NOISE] And the reason we have a random function here is because if you don't initialize your weights randomly, you will end up with some problem called the symmetry problem where every neuron is going to learn kind of the same thing. To avoid that, you will make the neuron starts at different places and let

[1:04:34] them evolve independently from each other as much as possible. So now we have two choices. Either we go over regularization or optimization. How much have you talked about regularization so far L1, L2, early stopping, all that? Early stopping, everybody remembers what it is? No? Little bit? So let's go over optimization, I guess, and then we will do some regularization depending on the time we have. [NOISE]

[1:05:09] So I believe so far you've seen gradient descent and stochastic gradient descent as two possible optimization algorithms. In practice, there is a trade-off between these two which is called mini-batch gradient descent. What is the trade-off? The trade-off is that batch gradient descent is cool because you can use vectorization, you can give a batch inputs, forward propagate it all at once doing vec- using a vectorized code. Stochastic gradient descent's advantage is that the updates are very quick.

[1:05:40] And imagine that you have a dataset with one million images. One million images in the dataset and you wanna do batch gradient descent. Do you know how long it's going to take to do one update? Very long. So we don't want that because maybe we don't need to go over the full dataset in order to have a good update. Maybe the updates based on 1,000 examples might already give us the right direction for the gradient [NOISE] of where to go. It's not gonna be as good as on the median example where it's going to be a very good approximation. So that's why most people would use mini-batch gradient descent,

[1:06:11] where you have a trade-off between stochasticity and also vectorization. So in terms of notation, [NOISE] I'm going to call X the matrix x_1, x_2, x_m, and capital Y the same matrix with y_m. So we have m training examples. And I'm going to split these into batches.

[1:06:44] So I'm going to call the first batch x_1 like this until x maybe T like that. And x_1 can contain probably x_1 until x_1,000. Assuming it's a batch of 1,000 examples. X_2 then will contain x_1,001 until x_2,000 and so on. So this is the notation for the batch when I use curly brackets.

[1:07:15] Same for Y. [NOISE] So in terms of algorithm, how does the Mini-batch gradient descent algorithm work? We're going to iterate. So for iteration t from 1 to blah, blah, blah,

[1:07:50] to how many iteration you wanna do. We're going to select a batch, select a batch of x_t- x_t, y_t. You will forward propagate the batch, and you will backpropagate the batch.

[1:08:21] So by forward propagation, I mean, you send all the batch to the network and you compute the loss functions for every example of the batch, you sum them together and you compute the cost function over the entire batch, which is the average of the loss functions. And so assuming- assuming the batch is of size 1,000, this would be the- the formula to compute the batch over 1,000 examples.

[1:09:01] And after the backpropagation, of course, updates, W_l and D_l for all the l's, for all the layers. This is the- the equation. So in terms of graph,

[1:09:36] what you're likely to see is that for batch gradient descent, your cost function j would have looked like that, if you plot it against the number of iterations. On the other hand, if you use a Mini-batch gradient descent, you're most likely to see something like this. So it is also decreasing as a trend, but because the gradient is approximated and doesn't necessarily go straight to the- to the middle of

[1:10:07] your loss fun- to the lower point of the loss function, you will see a kind of graph like that. The smaller the batch, the more stochasticity. So the more noise you will have on your cost function graph. And of course, if you- if we plot again- if we plot the loss function and this was gradient descent,

[1:10:37] so this is the top view of the loss function, assuming we're in two dimensions. Your stochastic gradient descent or batch gradient descent would do something like that. So the difference is- there seem to be less iteration with the red algorithm, but the iterations are much heavier to compute. So each of the green iterations are going to be very- very- very quick, while the red ones are going to be slow to compute. This is a trade off.

[1:11:11] Now there is another algorithm that I wanna go over which is called the momentum- momentum algorithm. Sometimes called gradient descent plus momentum algorithm. So what's the intuition behind momentum? The intuition is, let's look at this loss contour plot.

[1:11:49] And I'm doing an extreme case just to illustrate the intuition. Assume you have the loss that is very extended in one direction. So this direction is very extended and the other one is smaller. You're starting at a point like this one. Your gradient descent algorithm itself is going to follow the falling bar, it's going to be orthogonal to the current contour,

[1:12:21] uh, iso- iso term. Contour loss is going to go there, and then there, and then there, and then there, and so on. So what you would like is to move it faster on the horizontal line and slower to the vertical- on the vertical side. So on this axis you would like to move with smaller updates. And on this axis, you wanna move with larger updates, correct?

[1:12:51] If this happened, we would probably end up in the minimum much quicker than we currently are. So in order to do that, we're going to use a technique called momentum, which is going to look at the past gradients. So look at the past updates. Assume we're here. Assume we are somewhere here. Gradient descent doesn't look at its past at all. You just will compute the forward propagation, compute the backdrop, look at the direction and go to that direction. What momentum is going to say is look at the past updates that you did

[1:13:22] and try to consider these past updates in order to find the right way to go. So if you look at the past update and you take an average of the past update. You would take an average of these update going up and the update after it going down. The average on the vertical side is going to be small, because one went up, one went down. But on the horizontal axis, both went to the same direction. So the update will not change too much on the vert- on- on this axis. So you're most likely to do something like that if you use momentum.

[1:14:01] Does it make sense the intuition behind it? So that's the intuition why we want to use momentum. And for those of you who do physics, sometimes you can think of momentum as friction. You know like- like if you- if you launch a rocket and you wanna move it quickly around. It's not gonna move, because the rocket has a certain weight and has a certain momentum. You cannot change its direction very, very noisily. [NOISE]

[1:14:38] So let's see the implementation of- of- of momentum gradient descent. Oh, and I believe we- we're almost done, right? Yeah. Okay. [NOISE] So let's look at the- the implementation quickly. So gradient descent was w equals w minus Alpha, derivative of the loss with respect to w. What we are going to do is we're going to use another variable called velocity, which is going to be the average of the previous velocity and the current weight updates.

[1:15:18] So we're going to use that, and instead of the updates being the derivative directly, we're going to update the velocity. So the velocity is going to be a variable that tracks the direction that we should take regarding the current update and also the past updates with a factor Beta that is be- going to be the weights. The interesting point is that in terms of implementation it's one more line of code, in terms of memory,

[1:15:48] it's just one additional variable, and it actually has a big impact on the optimization. There are much more optimization algorithms that we're not going to see together today. In CS230, we teach something called RMSProp and Atom. That are most likely the- the- the ones that are used the most in deep learning. Uh, and the reason is, uh, if you come up with an optimization algorithm, you still have to prove that it works very well on the wide variety of application between- before researchers adopt it for their research.

[1:16:21] So Atom brings momentum to the deep learning optimization algorithms. Okay. Thanks guys. Uh, and that's all for deep learning in CS229 so far.
