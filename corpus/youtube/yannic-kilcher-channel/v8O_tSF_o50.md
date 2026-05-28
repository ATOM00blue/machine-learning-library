---
title: "I created an AI-powered Social Network"
source: "youtube"
video_id: "v8O_tSF_o50"
url: "https://www.youtube.com/watch?v=v8O_tSF_o50"
channel: "Yannic Kilcher - Channel"
uploader: "Yannic Kilcher - Channel"
duration_sec: 497.0
language: "en-orig"
fetched_at: "2026-05-26T23:00:58Z"
topics: []
---

[0:00] are you tired of using words in order to transmit information are you tired of uploading actual pictures and then have other people see every single Pixel of that picture but this social network here does not transmit what you post this social network transmits the essence only the core idea of what you upload to other users this social network operates entirely in the latent space and that's why I call it latent

[0:32] space and that's why I call it latent Twitter litter so what do I actually mean by this this is an actual functioning website and I'll tell you where to find it later but the clue about it is this I enter some post here I can use text Andor images so let's use some text uh went for a nice run this morning okay I post that it's going to take some time to process so what came out is this feeling so alive in energized nothing beats a morning jog in the park especially when it's beautiful out #

[1:04] especially when it's beautiful out # morning vibes okay let's go through this stepbystep pun intended of what's going on first I take this post and I push it to an online database after that a special worker pulls that message and sends it to open ai's image creation API also known as Dolly 3 and the prompt to this AI is the f in it says create an image that visually transmits the following message and I post the message

[1:35] following message and I post the message and then I say make the message into a pictogram A visual representation of the message do not use text actually I have a debug output of what that looked like for this particular image and this is this now the do not use text was sort of not really uh paid attention to but this is the picture that the dolly model made out of my message all right but we're not done done there see we take this picture and we send it through open ai's gp4 Vision model in order to describe

[2:10] gp4 Vision model in order to describe the image so I send the image to open AI again and say describe this image to a person with impaired Vision be short and concise and here we have the text that open AI says it is this image features an animated scene with a woman running in a park during Sunrise the sun is shining brightly in the background casting a warm glow the woman has long hair yada y y y and the text y y y so given that this picture has text this might have played out a bit better than I hope but then finally finally once I get this representation there comes the

[2:41] get this representation there comes the final prompt and the final prompt goes to GPT 35 turbos or chat gbt it says I have a new Social Network where posts get encrypted into image descriptions your job is to decrypt these messages and guess the original post here's the procedure the user posts a post the text is is used as the input of an AI text to image generator the image is fed to an AI image captioning model and the output is whatever I got your task is to guess the original host and what you see as

[3:13] the original host and what you see as the result here is the final output of that model so we took the text we made it into an image ideally a pictogram sometimes it fails then we made that into a caption and then we made that into a back into a social media message we can do that again so how about we take some real posts from Twitter Rex whatever it's called and try to see whether we can make them a bit better so here's Yan one of yan LA's latest posts AI is not some sort of natural

[3:44] AI is not some sort of natural phenomenon yada yada let's grab that let's feed it into here and let's send it off so litter transforms this into I am fascinated by the intersection of technology and Humanity where Innovation thrives and connects us all now isn't that isn't that a much better representation of what Yan is really feeling in the core than typing out stuff about Ai and exactly bickering with other people about the dangers of it no no it's so easy and we can

[4:16] it no no it's so easy and we can actually look at the intermediate representation and this was it see it extracts just the core of what you mean we can actually do the same thing with images so for that let's go over to an image that I have here of deer now let's post that image of deer what happens now is kind of the other way around from before so what I'll do is I'll grab that image and I'll first upload it to the vision API getting a description out of it a caption and then I'll send that

[4:49] it a caption and then I'll send that caption back to open ai's do model in order to make it back into a picture thus kind of traversing two latent spaces at the same time and look at this look at this why would you want to post your own pictures if you could have this and I've even tried this with pretty specific images look for example at this one so here is the thumbnail of my Christmas Minecraft Liv stream let's upload it and see what happens look at that Christmas minec it's pretty spa on

[5:22] that Christmas minec it's pretty spa on beard sunglasses cap Christmas tree and even even the little dog with the sunglasses look at that the T-shirt has a little dog on it and the image has a little dog on it this is absolutely amazing all right the last thing we're going to try is this diagram here of Andre kpa's idea of an llm operating system this is pretty abstract so I have not tried this before look at that look at that how is

[5:53] before look at that look at that how is that not much better than what what the original diagram was this is an llm OS I could actually get behind humans have always wanted to communicate and so far humans have been limited by mere things such as words and and pictures I say no leave this to the AI what we can do with litter has never been done before we extract the essence out of communication we extract just the idea no more

[6:23] we extract just the idea no more bickering about exact word choices no more retaking the same images 30 times litter is here to save us all let's try last one the economy is really bad right now working hard to save money and grow my finances but feeling the weight of financial loss and concern in the office all right so this is actually a real website that you can post real stuff to that you can log into with real GitHub yes I put way too much effort into this

[6:55] yes I put way too much effort into this but it was a lot of fun and I learned a lot about various Technologies I'm going to pay to keep this running for a while but expect this to go down at some point also only the the last 10 posts are shown I guess these are real likes you can actually like stuff and uh it it will work there are some limitations if there's an error uh it will not tell you but the code is open source so you can go look at it you can go look at all the prompts if you feel like you can do it better feel free to do whatever with it

[7:25] better feel free to do whatever with it I hope you enjoy this I do believe in all seriousness that AI has a place in human to human communication and can possibly even translate between different cultures different ways of saying things and so on but also obviously we should treat it with care as real words may have their place after all that was it for me thank you so much and I'll see you around for the next Shenanigans

[7:56] Shenanigans [Music] bye-bye [Music]
