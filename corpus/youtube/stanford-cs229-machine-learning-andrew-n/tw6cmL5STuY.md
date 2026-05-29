---
title: "Lecture 14 - EM Algorithm & Factor Analysis | Stanford CS229: Machine Learning Andrew Ng -Autumn2018"
source: "youtube"
video_id: "tw6cmL5STuY"
url: "https://www.youtube.com/watch?v=tw6cmL5STuY"
channel: "Stanford CS229 - Machine Learning (Andrew Ng, 2018)"
uploader: "Stanford CS229 - Machine Learning (Andrew Ng, 2018)"
duration_sec: 4788
language: "en-en"
fetched_at: "2026-05-26T22:52:14Z"
topics: []
aliases: ["Lecture 14 - EM Algorithm & Factor Analysis | Stanford CS229: Machine Learning Andrew Ng -Autumn2018"]
tags: [topic/classical-ml, level/intermediate, medium/lecture, task/tabular-classical]
---

[0:03] All right. Hi everyone, welcome back. Um, so what we'll see today is, um, additional, uh, elaborations on the EM, um, on the expectation maximization algorithm. And so, um, what you see today is, um, go over, you know, quick recap of what we talked about EM on Monday, and then describe how you can monitor if EM is converging.

[0:34] Um, and, um, uh, on, on Monday we talked about the mixture of Gaussians model, and started deriving EM for that. I want to just take these two equations and map it back to specifically the E and M steps that you saw for the mixture of Gaussians models, uh, to see exactly how these map to, um, uh, you know, updating the weights of the i and so on, um, how you actually derive the M step. Um, and then mostly what I want to spend today talking about is the model called the factor analysis model.

[1:06] Um, and this model useful for, um, for, for data, um, that can be very high-dimensional even when you have very few training examples. So what I wanna do is talk a bit about properties of Gaussian distributions, and then um, describe the factor analysis model, uh, some more about Gaussian distributions and then we'll derive EM for the factor analysis model. And, uh, I want to talk about factor analysis for two reasons, because one is, it's actually a useful algorithm in and of its own right. And second the derivation for EM for

[1:38] factor analysis is actually one of the trickier ones, and, uh, there are key steps in how you actually derive the E and M steps that I think you learn better or you better- master better by going through the factor analysis example. Okay. Um, so just to recap, last Monday or on Monday we had talked about the EM algorithm, uh, and we wound up figuring out this E-step and this M-step, right? And remember that if this is the log likelihood that you're trying to maximize,

[2:11] what the E-step does is it constructs a lower bound uh, that- this is a function of Theta. So this thing on the right hand side, this is a function of the parameters Theta. And what we proved last time was that, um, uh, that function is a lower bound of the log likelihood, right? And depending on what you choose for Q, you get different lower bound. So one choice of Q you may get this lower bound, for a different choice of Q you may get that lower bound. For a different choice of Q you may get that lower bound,

[2:41] and what the E-step does is it chooses Q to get the lower bound this tight, that just touches the log likelihood here at the current value of Theta, and what the M-step does is it chooses the parameters Theta that maximizes that lower bound, right? So that was the EM algorithm that we saw. Now um, I wanna step through how you would take this, you know, slightly abstract mathematical definition of EM and derive a concrete algorithm that you would implement, right? In, in, in, in, you know, um, in Python.

[3:14] And so let's, let's just step through this for the mixture of Gaussians model. Um, so for the mixture of Gaussians model we had a model for P of x i, z i which is P of x i given z i times p of z i, right? Um, and our model was that z is multinomial with some set of parameters phi, and so, you know, the probability of z i to be equal to j is equal to phi j, right?

[3:50] So phi is just a vector of numbers that sum to 1 specifying what is the chance of z being each of the, um, k possible discrete values. And then we have that x i given z i equals j, that, that is Gaussian with some mean and some covariance, right? And what we said last time was that, um, this is a lot like the Gaussian discriminant analysis model, uh and uh the, the, the trivial- one trivial difference is this is Sigma j instead of Sigma, right?

[4:22] GDA, Gaussian discriminant analysis, had the same Sigma every class but that's not the key difference. The key difference is that in, um, this density estimation problem, z is not observed or z is a latent random variable, right? Which is why we have all this machinery of, um, of EM. So now that you have this, um, uh, model, um, let's see.

[4:56] So now that you have this model, um, this is how you would derive the E and the M steps, right? So the E-step is, you know, you have Q i of z i, right? But let, let me just write this as Q i of z i equals j. Thi- this is sort of the probability of z i equals j. I know this notation's a little bit strange but under the Q i distribution, whether you want the chance of z being equal to j, right? And so, um, in the E-step you would say that the p of z i equals

[5:29] j given x i parameterized by all of the parameters. And we actually saw with Bayes' rule, right, how you would flesh this out, okay? And what we do in the E-step is, um, store this number, right? In, uh, what we wrote as w i j last time, okay? So you remember, um, if you have a mixture of two Gaussians, maybe that's the first Gaussian, that's the second Gaussian,

[6:00] you have an example x i here so it looks like it's more likely to come from the first than the second Gaussian and so this would be reflected in w i j. That, that example is assigned more to the first Gaussian than to the second Gaussian. So what you implement in code is, you know, you write code to compute this number and store it in wij. Um, and then for the M-step,

[6:35] you will want to maximize over the parameters of the model, right? Phi, mu, and Sigma, these are the param- uh pa-parameters of the mixture of Gaussians of sum over i, sum over z i, right? Um, and so the way you would actually

[7:08] derive this is you write this as sum of i. Um, so z i, you know, takes on a certain distribution of values. So z i we tu- turn, turn z i into j, right? So z I can be I guess one or two, if you have a mixture of two Gaussians. So you sum over all the indices of the different clusters of w i j times log of, uh, the numerator is, um, going to be

[7:43] into the negative one half times phi j, um, that's the numerator. And so, you know, this term is equal to, um, this first Gaussian term times that second term right, because this term is p of xi um, given z i, right?

[8:14] And the parameters, and this is just p of z i. Does that make sense? Okay. Um, and then we ta- take this and divide it by w i j, okay? So I'm, I'm going to step you through the, the, the steps you would go through if you're deriving EM using that, you know, E-step and M-step we wrote up above. But if you're deriving this for the mixture of Gaussians model then these are the, um, steps of algebra, right?

[8:46] You would, you would take, okay? Sorry I'm just realizing that. So in order to perform this maximization, what you will do is, um, you want to maximize this formula, right? This big double summation with respect to each of the parameters phi, mu, and Sigma. And so what you would do is, you know, take this big formula, right?

[9:16] And take the derivatives with respect to each of the parameters. So you take the derivative with respect to mu j, dot, dot, dot there's that big formula on the left, set it to 0, right? And take, uh, and then, and then it turns out if you do this, um, you will, uh, derive that mu j should be equal to sum over i to the i j x i over sum over i to the i j. And, um, this is what we said was how you update the mean's mu, right?

[9:50] The w i j's are the strength with which x i. So w i j is the, informally this is the strength with which x i is assigned, right? To Gaussian j, um, and more formally this is really p of, um, z i equals j given x i and the parameters, right?

[10:22] And so, um, so you end up with this formula. But the way you compute this formula is by the, the, the rigorous way to show this is the right formula for updating mu j, is looking at this objective, taking derivatives, saying they're zero, zero to maximize, um, and therefore deriving that equation for mu j, you know, by, by, by solving for the value of mu j that maximizes this expression, right? And, uh, similarly, you know, you take derivatives respect of, of, of this thing, with respect to that phi and set it to 0,

[10:54] take derivatives of this thing, um, right? And set that to 0 and that's how you would derive the update equations in the M-step for phi and for Sigma as well. Okay? Um, so and, and so, for example, when you do this, you find that the optimal value for phi is, um.

[11:27] Let's see. Yeah, yeah. We had this at the - near the start of Monday's lecture as well, okay? Um, so this is a process of how you would look at how the E-steps and M-steps are relative, and apply it to a specific model such as a mixtures of Gaussians model and that's how you, you know, solve for the maximization in the M-step, okay? And so what I'd like to do today is describe the application of EM.

[11:57] It's a more complex model called the factor of analysis model, and so it's important that - so I hope you understand the mechanics of how you do this, because we're going to do this today for a different model, okay? Any questions about this before I move on? Okay. Cool. Oh, so in order to,

[12:32] you know, foreshadow a little bit what we'll see when it comes down to the mixture of Gaussians model, excuse me. The Factor Analysis model which we talked about, you know, which is what we'll spend most of the day talking about. In the factor analysis model, instead of zi being discrete, zi will be continuous, right? And the particular zi will be distributed Gaussian. So the mixture of Gaussians model we had a joint distribution for x and z, where x was a discrete random variable. So in the factor analysis model we'll,

[13:03] we'll describe a different model. You know, for p of x and z, where z is continuous. And so instead of sum over zi, this would be an integral over zi of dzi, right? So - so sum becomes an integral. And - and it turns out that, yeah. Well, right. Yeah. And - and it turns out that if you go through the derivation of the EM algorithm that we worked out on Monday, all of the steps with Jensen's inequality,  all of those steps work exactly as before.

[13:37] Meaning you check every single step for whether zi was continuous it'll work the same as before if you have changed the sum to an integral, okay? All right. So let's see.

[14:11] So I want to mention one other view of EM that's equivalent to everything we've seen up until now which is, let me define j of theta, Q as

[14:43] this, okay? So is that formula you've seen a few times now. What we proved on Monday, was L of theta is greater than or equal to J of theta, Q right? And this is true for any theta and any choice of Q, okay? So using - using Jensen's inequality, you can show that, you know, J for any choice of theta and Q is a lower bound for the log likelihood of theta.

[15:19] So it turns out that an equivalent view of EM as everything we've seen before, is that at an E-step, what you're doing is maximize J with respect to Q and in the M-step maximize J with respect to theta, right? So in the E-step you're picking the choice of Q that maximizes this,

[15:52] and it turns out that the choice of Q we have we'll set J equal to L, and then in the M-step maximize this with respect to theta and pushes the value of L even higher. So this algorithm is sometimes called coordinate ascent. If you have a function of two variables and you optimize with respect to this, and also with respect to this, then go back and forth and optimize with respect to one at a time. That - that's the procedure that sometimes we call coordinate ascent, because you're maximizing with respect to one coordinate at a time. And so EM is a coordinate ascent algorithm relative to this course function J, right?

[16:27] And - and, you know, and every iteration J ends up being sent to L which is why you know that as the algorithm increases J, you know that the log-likelihood is increasing with every iteration and if you want to track whether the EM algorithm is converging or how it is converging, you can plot, you know, the value of J or the value of L on successive iterations and see if its validated, scoring monotonically and then when it plateaus and isn't improving anymore, then you might have a sense that the algorithm is converging, okay?

[17:04] All right. Okay. So that's it. The, basically an algorithm and a mixture of Gaussians. What I want to do now is - and is going to talk about the factor analysis algorithm. All right.

[17:39] So you know, that the factor analysis algorithm will work, actually sorry. So I want to compare and contrast mixture of Gaussians with factor analysis we're talking about a little bit, which is - for the mixture of Gaussians, let's say n equals 2 and m equals 100, right? So you have a data set with two features x 1 and x 2. So n is equal to 2 and maybe you have a data set that looks like this.

[18:11] You know, there's a mixture of two Gaussians. We have a pretty good model for this data set, right? Can fit one Gaussian there, fit the second Gaussian here. You kind of capture a distribution like this with a mixture of two Gaussians. And this is one illustration of when, when you apply mixture of Gaussians in this picture, m is much bigger than n, right? You have a lot more examples than you have dimensions. Where I would not use mixture of Gaussians

[18:45] and where you see the minute factor analysis will apply. Maybe if m is about similar to n, I don't know, even n is - or even m is much less than n, okay? And so just for purpose of illustration let say m equals 30 and n equals 100, right?

[19:18] So let's say you have 100 dimensional data but only 30 examples. And so to - to make this more concrete, you know, many years ago there was a Stanford PhD student that was placing temperature sensors all around different Stanford buildings. And so what you do is model, you measure the temperature at many different places, right? Around campus. But if you have 100 sensors,

[19:52] you know, taking 100 temperature readings around campus. But only 30 days of data or maybe 30 examples, then you would have 100 dimensional data because each example is a vector of 100 temperature readings, you know, at different points around this building say. But you may have only 30 examples of - of - if you have say 30 - 30 such vectors. And so the application that the Stanford PhD student at the time was working on,

[20:23] was she wanted to model p of x, right? So this is x as a vector of 100 sensors, 100 temperature readings. Because if something goes wrong or for example because a bad case would be if there's a fire in one of the rooms, then there'll be a very anomalous temperature reading in one place. And if you can model p of x and if you observe a value of p of x that is very small. You would say oh it looks like this anomaly there, right? And we're actually less worried about fires on Stanford.

[20:53] The use case was actually a - a- was it energy conservation. If someone unexpectedly leaves the window open in the building you are studying, you know, and it was hot and was it, and it's winter and it's warm inside the building and cool air blows in, and the temperature of one room drops in an anomalous way, you want to realize if something was going wrong with the windows, or the - or the temperature in part of the building, okay? So for an application like that, you need to model p of x as a joint distribution over,

[21:24] you know, all of the different senses, right? If you imagine maybe just in this room, let say we have 30 sensors in this room, then the temperatures at the 30 different points in this room will be highly correlated with each other. But how do you model this vector of a 100 - 100 dimensional vector with a relatively small training set? So it turns out that the problem with applying a Gaussian model. Well, all right.

[22:01] So one thing you could do is model this as a single Gaussian. And say that x is distributed, right? And if you look in your training set of 30 examples and find the maximum likelihood estimate parameters, you find that the maximum likelihood estimate of mu is just the average. And the maximum likelihood estimate of sigma is this,

[22:35] but it turns out that if m is less than equal to n, then sigma, this covariance matrix will be singular. And singular just means, uh - non-invertible, okay? I'll set for another illustration in a second.

[23:15] But, er, if you look at the formula for the Gaussian density, right? So the Gaussian density kind of looks like this, right? Abstracting away some details. And when a covariance matrix is singular, then this term, this determinant term will be 0. Um, so you end up with one over 0. Um, and then sigma inverse is also undefined or,

[23:49] er, blows up to infinity it will depending on how you think about it. Right but so, you know the inverse of a matrix like, um, 1, 10, right? Would be I guess one, 1 over 10, right? And, er, an example of a non-invertible matrix or singular matrix would be this, and you can't actually calculate the inverse of that matrix, right? So it turns out that, um, if your number of training examples is less than the dimension of the data, if you use the usual formula to derive the maximum likelihood estimate of Sigma,

[24:21] you'll end up with a covariance matrix that is singular. Uh, singular just means non-invertible, which means our covariance martix looks like this. And so, you know, the Gaussian density, if we try to compute p of x you get- you kind of get infinity over 0. You see, right? Oh, sorry not infinity, actually 0 over 0. Sorry, right. It doesn't matter, it's all bad. Um, and I think- let me just illustrate what this looks like. Which is, um, let's say m equals 2,

[24:53] and n equals 2, right? So you have two-dimensional data x1 and x2, and, um, uh, so n equals 2, and the number of training examples is equal to 2. So it turns out that, um- let's see, so you see me draw contours of Gaussian densities like this, right? Like ellipses like that. It turns out that if you have two examples, a two-dimensional space, and you compute the most likely- maximum likelihood estimate of the parameters of the Gaussian to fit to this data, then it turns out that these contours will look like that, right?

[25:24] Um, except that, instead of being very thin, as I'm drawing it, it will be, it will be infinitely skinny. So you end up with a Gaussian density where I can't draw lines, you know, of 0 width on the whiteboard, right? Um, uh, but it turns out that the contours will be squished infinitely thin. So you end up with a Gaussian density all- all of whose mass is on the straight line over there with infinitely thin contours that, that they're just, you know, we squish the centers on the, on the plane that goes on the line,

[25:56] um, connecting these two points. And so this is- so first there are, uh, practical numerical problems, right? As you end up with 0 over 0 if you try to compute p of x for any example. And second, um, this is- this very poorly conditioned Gaussian density puts all the probability mass on this line segment and so any example, right? Over there, just a little bit off, has no probability mass because, oh, has a probability mass of 0, a probability density of 0 because the Gaussian is squished infinitely thin,

[26:27] you know, on that, on that line, okay [NOISE]. But, but you can tell, this is just not a very good just- this is not a very good model, right? For, for this data. Um, So what we're gonna do is, ah, come up with a model that will work even for, um, for these applications, even, even for a dataset like this, right? Um, there's actually a, uh- I think the, the origins of the factor analysis model, uh,

[26:58] one of the very early applications was actually a psychological testing. Um, where, uh, if you have a, you know, administer a psychology, um, ah, exam to people to measure different personality attributes, right? So you might measure- you might have 100 questions or measure 100, uh, psychological attributes, um, but have a dataset of 30 persons, right? And again, you know, doing, doing psych research,

[27:30] collecting, you know, assembling survey data is hard. We assume you have a sample of 30 people and each person answers 100 quiz questions. Um, and so each person is one- gives you one example, right? X, and the dimension of this is, um, 100 dimensional, we have only 30 of these. And so if you want to model p of x, try to model how correlated are the different psychological attributes of people, right? Oh, is intelligence correlated with math ability, is that correlated with language ability, is that correlated with other things,

[28:02] uh, then how do you build a model for p of x, okay? All right. So, um, if the standard Gaussian model doesn't work, let's look at some alternatives. Um, one thing you could do is, uh, constrain Sigma to be diagonal, right?

[28:33] So Sigma is a covariance matrix, is an n by n covariance matrix. So in this case, it would be a 100 by 100 matrix. Um, but let's say we constrain it to just have diagonal entries and 0s on the off diagonals, right? So these giant 0s, I mean, the diagonal entries of this square matrix are these values in all of the entries of the diagonals you set to 0. So that's one thing you could do. And this turns out to be, um- this turns out to correspond to

[29:04] constraining your Gaussian to have axes align contours. So this is a Gaussian with 0 off-diagonals. Um, this would be another one, right? This would be another one. So these are examples of Gaussian- of, of contours of Gaussian densities with, um, 0 off diagonals. So the axes here are the X1 and X2, right? Whereas you cannot model something like this if your off diagonals are, are 0.

[29:39] Um, and so you do this, the maximum likelihood estimate of the parameters Sigma j, is pretty much what you'd expect actually. Right. The maximum likelihood estimate of the mean vector mu is the same as before. And this is maximum likelihood estimate of Sigma j, right? This kind of knowledge should be no surprise, it's kind of what you'd expect. Uh, and it turns out that, uh, right- and, and so the covariance matrix here has n parameters,

[30:09] instead of n squared or about n squared over two parameters, the covariance matrix Sigma now just has n parameters, which is the n diagonal entries. Now, the problem with this is that, this modeling assumption assumes that all of your features are uncorrelated, right? So you know, this just assumes that any two features they kind of share are, are completely uncorrelated. And, um, if you have temperature sensors in this room, it's just not a good assumption to assume

[30:39] the temperature at all points of this room are completely uncorrelated, completely independent of each other, or if you measure, you know, psychological attributes of people, it's just not a great assumption to assume that, you know, the different- different psychological measures you might have are completely, um, uh, independent. So while this model would take care of the problem, the, the technical problem of the covariance matrix being singular you can fit this model [NOISE], um you know, on a, on 100 dimensional dataset with 30 samples. You can fit this, you won't get this- you could build this model,

[31:12] you won't run into numerical singular, um, covariance matrix type problems, it's just not a very good model where you're just assuming nothing is correlated to anything else [NOISE]. Something else that you can do is, um, uh, make an even stronger assumption.

[31:44] So this is an even worse model, but I want to go through it because it will be a building block for what we'll actually do later, which is constrain Sigma to be Sigma equals, um, lowercase Sigma squared times i, right? And so, um, constrain Sigma to be dia- not only diagonal but to have the same entry in every single element. So now you've gone from, um,

[32:16] I guess n parameters to just one parameter, right? Uh, and this means that you are constraining the covariance matrix to- you are constraining the Gaussian you use to have circular contours. So this is an example where you can model. Uh, and this would be another example, right? And this is- I guess this is another example, okay? So you can model things like this, where every feature, not only is every feature uncorrelated but every feature further has the same variance as every other feature. Um, and the maximum likelihood is

[32:49] this, okay? And again, not, not, not a huge surprise, just the average over, uh, the previous values. So what we'd like to do is, um, not quite use either of these options, right? Which assumes- really, the biggest problem is it assumes the features are uncorrelated. Um, and what I'd like to do is build the model that you can fit even when you have very high dimensional data and a relatively small number of examples,

[33:23] um, but that allows you to capture some of the correlations, right? So if you have 30 temperature sensors in this room, you know, probably there are some correlations, right? Probably, this side of the room temperature is gonna be correlated, and that side of the room temperature is gonna be correlated and maybe the ambient temperature in this whole building. The, the temperature of this room really goes up and down as a whole, but maybe some of the lamps on the side heat up that side of the room a bit more, so different, the different. There are correlations but maybe you don't need a full covariance matrix either. So what [NOISE], what factor analysis will do is, um,

[33:54] give us a model that you can fit even when you have, you know, [NOISE] 100 dimensional data and 30 examples. They capture some of the correlations but that doesn't run into the a, a- uninvertible, um, covariance matrixes is that the naive Gaussian model does, okay? All right. So let me- just check any- let me, let me describe the model, let me just check,

[34:24] any questions before I move on? Okay. [BACKGROUND] Oh, sure. Yes. Um, yes. There is one thing you can do. A common thing to do is apply Wishart prior and what that boils down to is, um, add a small diagonal value to that- to the maximum likelihood estimate. Um, it- it kind of, uh, in a technical sense it takes away the, uh, non-invertible matrix problem. Uh, it's actually not the best algorithm for a lot of the types of data.

[34:57] Um, uh, the- the- the- the Wishart or inverse Wishart prior, yeah. Others, you know- basically, take the maximum likelihood for Sigma, and add, you know, some constant to the diagonal. Um, it takes care of the problem in a technical way, but it is not- it's not the best model for a lot of datasets, I see. Why do we even think about option two [inaudible] Oh, yes. Why do you think about option two, when it's likesemi even worse than option one. Um, yes.

[35:28] Option two is not a good option, but I need to use this as a building block for factor analysis. So you see this is a small component of, uh, of, uh, see I actually planned these things out, you know. [LAUGHTER]. Cool, yeah. And- and- and maybe- actually to- to- to give [inaudible] just- just to mention, you know, um, just mention some things I see. Yeah. Actually the- the machine learning work evolves all the time, which I find fascinating. But you look at all the big tech companies, um, a lot of the large tech companies, they're all like working on exactly the same problems, right? Every large tech company, you know,

[35:58] software, AI company, is working on machine translation, every one of them works on speech recognition, every one of them works on face recognition, and I- I- I've been part of these teams myself. Right? And I think it's great that we have so much progress in machine translation, because there are so many people, and so many large companies that work on machine translation. It's actually really happy to see so much progress in these problems that every single large tech company, large software, AI-ish tech company works on. Um, one of the fascinating things I see is that, um, uh, because of all this work into large tech companies working on very similar problems,

[36:32] one of the really overlooked parts of the machine learning world is small data problems, right? So there's a lot work in big data if you are Brazilian, English, and French, and Chinese, and Spanish sentences is the semi-close models that work. Um, and I think, uh, uh, there's actually a lack of attention, like a disproportionately small amount of attention, on, you know, small data problems, where instead of, uh, 100 million images, you maybe have 100 images. Um, and so, uh, some of the teams I work with these days, actually like Landing AI. Um, I actually spent a lot of my time thinking about small data problems,

[37:05] because a lot of the practical applications of machine learning, including a lot of things you see in your class projects, are actually small data problems. Right? And I think, um, when- when Annan, uh, worked with a healthcare system, works at Stanford Hospital, for some of the problems, you only have 100 examples, or even 1,000, or even 10,000. You don't have a million patients with the same medical condition. And so I think that, um, uh, a lot of these models- So- and again, uh, earlier this week, I was using a slightly modified version of factor analysis on a manufacturing problem at Landing AI.

[37:38] Right? And I think a lot of these small data problems are actually where a lot of the exciting work is to be done in machine learning, and is somehow- it- it feels like a blind spot of- or if like a- like a- like a gap of, uh, a lot of the work done in the AI world today. Go ahead. [inaudible]. Uh, yeah. Why don't we use the same algorithms with this big data?

[38:10] It turns out that, um, uh, you know, it turns out- if- if- if you look at the computer vision world, right? There's a data set that everyone is working on. Now- now we're past it, we don't really use it any more, called ImageNet, which had a million images, and so there are tons of computer vision architectures that have been heavily designed for the use case of if you have exactly one million training examples. Uh, and it turns out that the algorithms that work best if you have 100 training examples is, you know, looks like it's different than the best learning algorithm. I think, um, uh,

[38:40] uh, and so I think right now, we actually- I think the machine learning world, we are not very good at understanding the scaling. Uh, the best algorithm for one training example, you know, as far as we are able to invent algorithms as a community, is different than best algorithm for 1000, best for, for a million, it's actually different than, um, uh, uh- actually, and Facebook published a paper recently, with 3.5 billion images. The result was cool, it was very large, right? So I was saying, we don't actually have a good understanding of how to modify our algorithms, to have one algorithm work on every single point of this spectrum,

[39:13] going from one example to, like, a billion examples. Uh, and so there's a lot of work optimizing for different points of the spectrum, uh, and I think there's been, um, a lot of work optimizing for big data, which is great, you know, build some of these large systems that handle, like, whatever, petabytes of data a day, uh, that's great. But, um, uh, I feel like relative to the number of, um, application opportunities, there- there's a lot of work on small data well, that- that I find very exciting, that- that, uh, and I think of this as an example.

[39:44] Uh, the reason I was using this, literally, well, modified version of this, earlier this week on the manufacturing problem, um, is because, um, uh, there isn't that much data in those scenarios, right? Cool. All right. That's, um, off-topic. But let's- let's- let's go and describe- well, hopefully, maybe so, so this stuff does get used, right? Uh, so let's- let's talk about the model. Um, so similar to, uh, the mixture of Gaussians, I'm gonna define a model with,

[40:16] um, P of X, Z equals P of X, given Z times P of Z, uh, and Z is hidden. Okay? So that's the framework, same as, um, mixture of Gaussian. So let me just define the factor analysis model.

[40:49] So first, um, Z will be drawn- distributed according to the Gaussian density, where Z is going to be an RD, where D is less than N. And again, to think about it, um, maybe you can think of it as, uh, D equals 3, uh, uh, N equals 100, M equals 30. Okay? Um, and- and- but I guess, ju- just make sure this is a concrete example to think about it. And what we're going to assume is that X is equal to Mu,

[41:22] plus, um, Lambda Z. This is, uh, the capital Greek alphabet Lambda, plus Epsilon, where Epsilon is just using Gaussian with mean 0, and covariance Psi. Um, so the parameters of this model are Mu which is N dimensional, um,

[41:59] Lambda which is N by D, and Psi which is N by N, and we're going to assume that Psi is a diagonal. Okay? Um, and so- let's see. The second equation, an equivalent way to write that, equivalently, is that given the value of Z,

[42:29] the conditional distribution of X, right, X given Z, this is Gaussian with mean given by Mu plus, um, Lambda Z, and covariance Psi. Okay? So once you've given Z- once you sample Z- so this is P of Z and this is P of- P of Z and this is P of X, Z- X given Z. Right? So given Z, X is computed as Mu plus Lambda Z. So this is just some constant, and then you add Gaussian noise to it.

[43:00] And so this equation, an equivalent way to define this equation, is to say that the mean of X, uh, conditioned on Z, is this first term. Right? Since that's the mean. And the covariance of X given Z, is given by this, you know, additional term Psi, by that noise term that you add to it. Okay? So let me go through a few examples. And- and I think the intuition behind this model is, um,

[43:34] if- if you think that there are three powerful forces driving temperatures across this room, maybe one powerful force is just what is the temperature, you know, here in Palo Alto, what's the temperature here at Stanford. And another powerful force is how bright are the lights on the left side of the room, and how hot does it heat up this side of room, and another is how hot does is it heat up the right side of the room. Right? So, you know, let's say there are three main driving factors affecting the temperature of this room, then that's when D would be equal to 3. Then you assume that, you know, there are three things in the world that drive

[44:04] the temperature of this room that's three-dimensional, which is the temperature in Palo Alto, kind of, around this area, um, how bright that the light is there, and how bright that the light is there, and you try to capture that with three numbers. Given those three numbers, right? Given Z, the actual temperature for the 100 sensors we scatter around this room, will be determined by each sensor, right? So we plug 30 temperature sensors all over this room. Each sensor we plant will measure an actual temperature,

[44:35] that's a linear function of those three powerful forces, um, and if a sensor is on that side of the room, it'll be affected more by how bright that the lights are on that side of the room. Um, uh, if there's a sensor near the door, it will be more affected by the temperature outside- temperature here in Palo Alto. Right? But so X will be a linear function, but this first time I underlined. Um, but rather than just that term, there is [inaudible] noise. Right? So each sensor has its own noise term, which is governed by this additional noise term Epsilon.

[45:08] And, um, the assumption that this matrix Psi is diagonal, it's saying that after you compute the mean, the noise that you observe at each sensor is independent of the noise at every other sensor. Does that make sense? Right? That maybe- maybe the sensor, you know, up there, right? Maybe it's just noisy or something, just a gust of wind. But you assume that the noise of, you observe at different sensors is independent. The- the additional Epsilon error term has a-

[45:40] has a diagonal covariance matrix given by Psi. Okay? So you can- so you can think of that as what, um, uh, factor analysis is trying to model. Okay? So let me, um, just go through a couple of examples of the types of data factor analysis can model. All right, and again by the constraints of the whiteboard,

[46:11] I'm going to have to go low-dimensional here, right? Um, so actually let me- let me go through a couple examples. So let's say Z is R_1 and X is R_2. So in this example I guess d is equal to 1, n is equal to 2 and let's say m is 7, right? just- just. So what will be a typical example,

[46:42] generated by- what will be an example of a type of data that this can model? So this, let me erase this here. All right, so this would be a typical sample of Z_i right? which is you know- so this is z is just drawn from a standard Gaussian. So I guess z is just Gaussian, would mean 0 and unit variance.

[47:12] So that's the number line and you draw seven points from a Gaussian, you know, maybe you get a sample like that. Okay? and now let's say lambda is 2, 1 and let's just say mu is 0, 0, okay? So now let's compute lambda x plus mu, right?

[47:44] so given a typical sample like that um, if you compute lambda x plus mu, this will now be the R_2, right? so here is X_1, here is X_2. We're gonna take those examples and map them to a line as follows. Where these examples on R_1. So- excuse me, lambda z plus mu, okay. So this is just a real number and so lambda z plus mu is now two-dimensional,

[48:17] right? Because lambda is a 2 by 1 matrix. Okay? so you end up with- So this would be a typical sample- typical random sample of lambda z plus mu and it's a two-dimensional data-set but all of the examples lie perfectly on a straight line. Okay? Then finally let's say that psi, the covariance matrix is equal to this as a diagonal covariance matrix and

[48:48] so this covariance matrix corresponds to X_2 having a bigger variance than X_1, right? And so you know this, this- I guess the density of epsilon has ellipses that look a little bit like this, it's taller than wide. The aspect ratio should technically be 1 over root 2 to 1, right? Because the standard deviations will be root 2, I guess. And so in the last step of what we are going to do, x equals lambda z plus mu plus epsilon. We're going to take each of these points we have and put

[49:19] a little Gaussian contour. You know there's that shape. There's this- I'm just drawing one contour of the shape and just put it on top of this, and if you sample one point from each of these Gaussians, then maybe you get this example, this example, this example, this example, okay? So what I just did was look at each of the Gaussian contours and sample a point from that Gaussian. And so the red crosses here are a typical sample drawn from this model.

[49:55] Okay? and so if you have data that looks like this, that looks at the red crosses. The Zs are latent random variables, right? When you get the dataset you kind of just see Zs. So what you actually see, is just you know the red crosses, that's your training set and if you apply the factor analysis model with these parameters then you can find EM and so on. Hopefully you can find parameters that models this dataset pretty well, but hopefully this gives you sense of the type of dataset this could generate and so- and so on.

[50:33] And one way to think of this data is you have two-dimensional data but most of the data lies on a 1D subspace. So this is how to think about it, you have two-dimensional data since n is two. But most of the data lies on a roughly one-dimensional subspace meaning it lies roughly on a line, and then there's a little bit of noise off that line, okay? All right, let me quickly do one more example because these are- these are high-dimensional spaces. I think it's- I think it's useful to build intuition.

[51:05] All right, so let's go through an example where z is in R_2, x is in R_3 and let's use m equals 5. So d equals 2, n equals 3, okay? So we have a different set of parameters. Let's look at the type of data you can generate a factor analysis which is, here is Z_1 and Z_2.

[51:37] Z is distributed Gaussian, standard Gaussian 2D so it would be a circular Gaussian. So maybe this is what the typical sample, right, looks like. If you- if you if you sample sort of Z_1 and Z_2 from a standard Gaussian, right that would be a typical sample in Z_1 and Z_2. So now- all right, I'm going to do a demo. Let me take these five examples and just copy them to this piece of paper, okay? So, all right there, right?

[52:13] Transferred it from the whiteboard to this piece of paper, to this brown cardboard. So now you have Z_1 and Z_2 in a two-dimensional space. What we're going to do is compute lambda z plus mu, and this will be 3 by 2, and this will be 3 by 1. So what this computation will do as you map from z in two-dimensions to lambda z plus mu, is you're going to map from two-dimensional data to three-dimensional data.

[52:45] In other words, you want to take the two-dimensional data lying on the plane in the whiteboard, and map it, check out this cool animation into the three-dimensional space of our classroom [LAUGHTER]. And then the last step is for each of these points in this three-dimensional space like X_1 X_2 X_3, right? We'll have a little Gaussian bump that is axis

[53:16] aligned because epsilon is the features, the- the components of epsilon are uncorrelated and taking each of these five points and add a little bit of fuzziness, add a little bit of Gaussian noise to it. And so what you end up with is a set of red crosses and you end up with a few examples, you know add a little bit of noise, you end up with- except that they would have a bit of noise off this plane as well, right? But so what the factor analysis model can capture is if you have data in 3D, right?

[53:50] In this 3D space, but most of the dataset lies on this maybe roughly two-dimensional pancake but there's a little bit of fuzziness off the pancake, right, so this would be an example of the type of data that factor analysis can model. Okay? All right cool. Um, and the intuition is really think of factor analysis can take very high dimensional data, say, 100 dimensional data and model the data as roughly lying on a three-dimensional,

[54:20] five dimensional subspace with a little bit of fuzz, with a little bit of noise off that low dimensional subspace. Great. So- [NOISE]

[54:54] All right. So let's talk about- yeah. [BACKGROUND] Oh, right. It does not work as well if the data's not lying on low dimensional subspace. Um, let's see. So even in 2D, if you have, um, this data set, right? [NOISE] You actually have the freedom to choose Gaussian noises like that, in which case you can actually model things that are quite far off a subspace. Uh, but, um, uh, yeah, I, I, I, you know,

[55:26] I think when you have a very high dimensional data set, it's actually very difficult to know what's going on because you can't visualize these very high dimensional data sets, uh, and you also don't have enough data to build very sophisticated models. So, so I feel like yes, if you have- if the data actually does not roughly lie in a subspace, then this model, you know, may not be the best model, but when you have such high dimensional data in such a small data set, um, you- is- you can't fit very complex models through it anyway, so this might be pretty reasonable.

[55:56] Right. Cool. All right. So, um- [NOISE] all right. So it turns out that the derivation of EM for factor analysis is actually, it's actually one of the trickiest EM derivations, in terms of how you calculate the e-step, and how you calculate the m-step. Um, the whole algorithm is, you know, describe the- every, every, every single step, the- step three in great detail in the lecture notes.

[56:28] But what I want to do is give you the flavor of how to do the derivation, and to especially draw attention to the trickiest step, so that if you need to derive an algorithm like this yourself for maybe a different Gaussian model, then you know how to do it, but I won't do every step of the algebra here. All right? Um, so in order to set ourselves up to derive factor analysis, uh, EN- ENM for factor analysis, I wanna describe a few properties of, uh, multivariate Gaussians. So [NOISE] let's say that X is a vector,

[57:00] and I'm gonna write this as a partition vector, right? In which, um, uh, [NOISE] if there are R components there, and S components there. So [NOISE] X_1 is in R_r, X_2 is in R_S, and X is in R_ r plus S. Okay? So if X is Gaussian with mean Mu and covariance Sigma, then, uh, let- similarly,

[57:31] let Mu be written as this sort of partition vector. Right? Just break it up into two sub-vectors, corresponding to the first R components in the second S components. And similarly, let the covariance matrix be partitioned into, um, you know, these four diagonal blocks, where, I guess, this is R components, this is S components, this is R components, this is S components. Um, so all this means is,

[58:02] uh, you take the covariance matrix, and take the top leftmost R-by-R elements, and call that Sigma 1, 1. Right? And, and, uh, and, and, and then similarly for the other sub-blocks of this, um, covariance matrix. So in order to derive factor analysis, one of the things you need to do is compute marginal and, um, uh, conditional distributions of Gaussians.

[58:33] So the marginal is, [NOISE] you know, what is P of X_1. Right? Um, and so the, the- if you, you know, were to derive this, uh, the way you compute the marginal is to take the joint density [NOISE] of P of X, right? And you can write this as P of X_1 X_2, because X can be partitioned into X_1 and X_2, and integrate out X_2 under P of X_1 X_2, right? Dx_2, and this will give you P of X_1.

[59:05] Right? And if you plug in the Gaussian density, the formula for the Gaussian density, if you plug in, I guess, you know, 1 over 2 Pi to the N over 2, is equals to one-half, right? E to the, you know, minus one-half, X1 minus Mu 1, X_2 minus Mu 2, uh, right?

[59:38] If you plug this into P of X_1, X_2, and actually do the integral, um, then you will find that, um, the marginal distribution of X_1 [NOISE] is given by; X_1 is usually a Gaussian, with mean Mu 1, and covariance sigma 1, 1. So it- it's, kind of, not a shocking result, that the marginal distribution is given just by that and that.

[1:00:11] Right? And, and again, the way to show it vigorously is to do this calculation, [NOISE] but it's actually not shocking, I guess, that that's what you would get. Okay? Um, and then the other property you will [NOISE] need to use is a conditional, which is, um, [NOISE] given the value of X_2, what is the conditional value of X_1? Um, and so the way to do that would be, you know, [NOISE] in theory, you would take P of X_1, X_2 divide by P of X_2,

[1:00:43] right? And then simplify. And it turns out you can show that, um, [NOISE] X_1 given X_2, is itself Gaussian, [NOISE] with some mean and some covariance, we're just gonna write this Mu of 1 given 2 and Sigma of 1 given 2, where Mu of 1 given 2 is, uh- and, and- but this is one of those formulas that I actually don't- I actually don't manage to remember, but every time I need it I just look it up. It's written in the lecture notes as well. So um, [NOISE] X_2

[1:01:20] minus 2 to- oops. Okay? [NOISE] So that's how you compute, um, marginals and conditionals of a Gaussian distribution. Okay? So [NOISE]

[1:01:57] using these properties of, uh, the multivariate Gaussian density, let's go through the high-level steps of how you derive the EM algorithm for this. [NOISE] All right. [NOISE] Um, step one is, uh, let's compute- actually, let's, um- excuse me.

[1:02:32] [NOISE] Let's derive what is the joint distribution of P of X and Z. Right? And in particular, it turns out [NOISE] that if you take Z and X and stack them up into a vector like so, um, Z and X viewed as a vector would be Gaussian with mean, um- [NOISE] with some mean and some covariance,

[1:03:04] uh, because X and Z jointly will have a Gaussian density. And let's try to quickly figure out what are this mean and that covariance matrix. [NOISE] So that was a definition of these terms. Um, and so the expected value of Z is equal to 0 because 0 is- Z is Gaussian with mean 0 and covariance identity,

[1:03:35] [NOISE] and the expected value of X is equal to the expected value of Mu plus Lambda Z, plus epsilon, um, but Z has 0 expected value, epsilon has 0 expected value, so that just leaves you with Mu. And so this mean vector Mu XZ, is going to equal to 0 Mu. Right. And so this is D-dimensional, [NOISE] and this is, uh, N-dimensional. Okay? Um, and it turns out that,

[1:04:08] uh, [NOISE] let's see- and it turns out that you can [NOISE] similarly compute the covariance matrix Sigma, right? Where this is, um, D dimensions and this is N dimensions.

[1:04:39] Um, [NOISE] it turns out that if you take this partition vector, and compute the covariance matrix, [NOISE] the four blocks of the covariance matrix can be written as follows. [NOISE] Um-

[1:05:16] Okay. And you can, one at a time, derive what each of these different blocks look like. Um, and let me just do one of these, and let me just derive what Sigma 2, 2, the lower right block is and the rest are derived similarly and also fleshed out in the lecture notes. So the way you derive what this block is like is that you say Sigma 2, 2 is x minus Ex,

[1:05:54] x minus Ex transpose. And so if I plug in the definition of x that would be a Lambda z plus Mu plus Epsilon minus Mu times the same thing. Right. Um, so there's x minus Ex. So there's x minus Ex, okay? Uh, because the expected value of x is Mu.

[1:06:27] So the Mus cancel out. And then if you do the quadratic expansion, I guess this becomes expected value of, um, let's see, Lambda z times each of these two terms transpose plus- it, it, it sort of, you know, a plus b times a plus b, right?

[1:07:01] It's a times a times a plus b, b times a, b plus b. You get four terms as a result. And so the first term is Lambda z times Lambda z transpose, which is this, plus Lambda z Epsilon transpose plus Epsilon, um, right? And so, um, this term has 0 expected value because,

[1:07:32] uh, Epsilon and, and z, both have zero expected value uncorrelated. So this is zero. This is zero on expectation. And so you're just left with the expected value of Lambda zz transpose, Lambda transpose plus the expected value of, uh, Epsilon Epsilon transpose, right? Um, and so by the linearity of expectation, you can take expectation inside a ma- matrix multiplication. So this Lambda times the expected value of zz transpose times Lambda transpose plus.

[1:08:07] And this is just the covariance of Epsilon, right? Which is- which is Psi. Um, and then because z is drawn from a standard Gaussian with identity covariance, that expectation in the middle is just the identity. So that's Lambda, Lambda transpose plus Psi. Okay. So that's how you work out what is this lower right block of this, um, covariance matrix. I know I did that a little bit quickly, but every, every step is,

[1:08:38] uh, written out, uh, more slowly in the lecture notes as well. Okay. And it turns out that if you go through a similar process to figure out, you know, one at a time using similar process, one of the other blocks of this covariance matrix, you find that the other blocks of this covariance matrix are identity, Lambda, Lambda transpose and the one we just worked out. Okay. That- that's the one we just worked out. But so that is the covariance matrix Psi. [NOISE]

[1:09:42] So where we are is that we've figured out that the joint distribution or the joint density of z x is Gaussian with mean given by that vector and covariance given by that matrix, okay? Um, and so what you could do, uh, is, um, you write down, right? P of x_i and try to take the- uh,

[1:10:12] so P of x_i will be this Gaussian density. And what you could do is take derivatives of the log likelihood with respect to the parameters, and set the parameters to 0 and solve. And you find that there is no known closed-form solution. There is actually no closed-form solution for finding the values of Lambda and Psi and Mu that maximize this log-likelihood. So in order to, uh, fit the parameters of the model, we're instead going to resort to EM, okay?

[1:10:46] And so in the E-step. Right. So let's, let's first derive what is the E-step, which is an E-step, you need to compute this, right? Now, um, z_i here is a continuous random variable. When we're fitting a mixture of Gaussian distributions, z_i was discrete,

[1:11:18] and so you could have a list of numbers represented by, you know, w_ i_ j, that just, just at the vector sorting what is the probability of each of the discrete values of z_i. But in this case, z_i is a continuous density. So how do you represent Qi of z_i at a computer? It turns out that using the formulas we have for the marginal- excuse me, for the conditional distribution of a Gaussian, it turns out that if you compute this right hand side, you find that z_i given x_i,

[1:11:50] this is going to be Gaussian with some mean and some covariance, right? Where- oh, it's basically those formulas, Mu of z_i given x_i is equal to- if you kinda of take that formula and apply it to our thing here, a zero, uh, plus Lambda transpose.

[1:12:29] And, um [NOISE] okay. So these equations exactly, these two equations, right, maps to- map to that big Gaussian Density that we have. Okay. So what you would do in the E-step is, um, compute this and compute this- compute this vector and compute this matrix,

[1:13:03] and store that- store these, you know, store these as variables, and your representation of the Q_i is that Q_i is a Gaussian Density, right, with this mean and this covariance. So this is what you actually compute to represent Q_i. Okay. [NOISE] All right.

[1:13:33] So step two was derive the E-step, and step three is derive the M-step. [NOISE] and, um, the derivation of the M-step is, is quite long and complicated, um, but I wanna mention just a key alge- algebraic trick you need to use when deriving the M-step. Um, so, you know, we know from the E-step that Q_i of z_i is that Gaussian Density.

[1:14:05] Right. So you know, it's 1 over 2 pi to the d over 2, that thing, and E to the, right, negative 1.5 dot dot dot. Right. So tha- that's the formula of a Q_i. It turns out that, um, in the M-step, there will be a few places in the derivation where you need to compute something like this. [NOISE] Right. And one way to approach this would be to plug into the density

[1:14:38] for Q_i which is [NOISE] So you'd end up with this. 1 over 2 pi to the d over 2 Sigma, you know, wha- uh, and so on into the negative 1.5 dot dot dot times Z_i d_Z_i, and then try to compute this integral. Um, it turns out there's a much simpler way to compute this integral. Anyone know what it is?

[1:15:13] All right. Cool. Awesome. Expected value. So the other way to compute this integral is to notice that this is the expected value of z_i when z_i is drawn from Q_i, right? So you know th- the, the definition of the expected val- value of a random variable is expected value of z is equal to integral over z probability_z times zdz, right? That's what the expected value of a random variable is.

[1:15:44] And so this integral is the expected value of z with respect to z drawn from the Q_i distribution. Um, but we know that Q_i is Gaussian with associated mean and certain variance, and so the expected value of this- this is just mu of z_i given x_i, right? It's that thing that you've already computed in the E-step. Makes sense? And so when students derive the M-step, you know, for EM implementations of Gaussians,

[1:16:14] one of the key things to notice is, uh, when are you actually taking an expected value with respect to a random variable, in which case, it's just the value computed already, and when do you need to plug in this big complicated integral which can lead to very complicated, very intractable calculations. Okay. So just when you're- whenever you see this, um, uh, think about whether you need to be expanding a big complicated integral, or if it can be interpreted as an expected value. Okay. Um, and so for the M-step,

[1:16:53] it's really, you know, the M-step is [NOISE] All right. So that's the M-step.

[1:17:25] And if you re-write this term as sum over i, the expected value of z_i, uh, drawn from Q_i of this- all right. It turns out that, um, if you go ahead and, uh, plug in the Gaussian density, here [NOISE] actually on-

[1:18:03] one rule of thumb for whether or not you should plug in a complicated integral or plug in a Gaussian density, um, this is just a rule of thumb after doing this type of math a long time, is that see if there's a log in front. If there's a log in front of a Gaussian density, basically Gaussian density has an exponentiation, right? The Gaussian density is 1 over e to the something. So whenever there's a log in front, the log exponentiation cancel out, and this equation simplifies. So one trick as you're doing these derivations is just see if there's a log in front of a Gaussian density. And when there is a plug in, go ahead and plug in the formula for your Gaussian density,

[1:18:35] the log will simplify that, and what you end up with is the log of a Gaussian density ends up being a quadratic function, a quadratic function of the parameters. And if you take the expected value with respect to a Gaussian density, respect to quadratic function, this whole thing ends up being a quadratic function. Um, and then you can take derivatives of that equation with respect to the parameters. With respect to mu of that whole thing, set it to 0,

[1:19:06] and then solve and they'll be roughly, um, level of complexity of, of maximizing quadratic function. Okay. Hope that makes sense. Um, the actual formulas are a little bit complicated. So I don't- I'll, I'll leave you to look at the actual formulas in the lecture notes, but I think the take away is, uh, don't expand this integral, um, and when you are deriving this, plug in the Gaussian densities here because the log will simplify. Okay. And details of it in the lecture notes. So let's break for today.

[1:19:36] Uh, best of luck with the mid-term and- seriously, I hope you guys do well. All right. I- I'll see you guys in a, in a few days.
