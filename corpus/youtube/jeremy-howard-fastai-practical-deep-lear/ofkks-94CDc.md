---
title: "Fast.ai APL study session 14"
source: "youtube"
video_id: "ofkks-94CDc"
url: "https://www.youtube.com/watch?v=ofkks-94CDc"
channel: "Jeremy Howard / fast.ai - Practical Deep Learning"
uploader: "Jeremy Howard / fast.ai - Practical Deep Learning"
duration_sec: 5281.0
language: "en-orig"
fetched_at: "2026-05-26T23:07:34Z"
topics: []
aliases: ["Fast.ai APL study session 14"]
tags: [topic/ml-engineering, level/intermediate, medium/lecture, task/general]
---

[0:24] hi hello hello hello how are you i'm good very good uh glad i made it how about you jeremy how are you doing huh oh can you hear me it's this it's the same thing again i know it's okay it's fine oh it's all good no how are you jeremy sweet sweet sweet uh it's beautiful and sunny yes beautiful

[0:55] beautiful like nearly every other day apparently it's going to be 27 which is pretty good for winter definitely not with the experience of my first 37 years of my life yeah not that's 37 this is sometimes closer to the distance from the zero on the number line on the other side you know

[1:27] line on the other side you know i mean ah i'm having trouble getting i just want to get my screen organized and um i'm uh not quite used to being on a mac there we go and i've also tried buying a um one of their magic trackpad things so

[1:59] one of their magic trackpad things so that takes some a surprising amount of getting used to as well so those are great things just take a little bit more adjusting to them what is it it's a truck part right yeah it's just the you know the magic track pads that they use on apple devices i mean it's just it's like that but i guess i'm so used to using a thumb trackball that i my brain's like what is this strange thing uh trackpad is the most hated piece of

[2:32] uh trackpad is the most hated piece of equipment for me of computer technology like uh you know i remember the thinkpads they had this tiny little like gizmo that you could uh use instead of the trackpad i'm not sure if you're here yes yeah that was wonderful and like i even hate using the mouse uh so so trackpad is like one level below buddy i'm also in the in this club after reading the

[3:02] after reading the rachel's post uh i made the jump and uh like i don't even think about it anymore it's uh yeah i had to change because my rsi was terrible um it made a big difference so i'm planning this may be crazy but i'm planning to like power through a lot of gifts today um so that's my hope

[3:36] um and i'll just also mention i guess i'll share my screen um oops so yeah i'm already pressing the wrong button on this track there i kind of got stuck on the end code which is the t-shaped glyph okay i guess we'll hopefully get to that today then uh hopefully we'll get to as many as possible um

[4:06] possible um all right i'm sharing my screen you can see it okay yep um just wanted to briefly mention i eyes oh that's not good yeah um i broke everything how about that your mic is not as good as the other day jeremy

[4:37] jeremy that's why on the other side of my computer thanks for letting me know okay is that better yes everybody else is too polite to say anything yes i people being polite is the bane of my existence nothing worse than being polite i didn't even notice it sounded fine to me so all right okay that's amazing that we get along jeremy then because i'm like chronically polite like uh i know you're disastrously

[5:08] like uh i know you're disastrously polite radically it is complicated um why on earth do we not have a website anymore um so gh pages so if you're ever trying to debug broken github pages if you actually go to the gh pages branch oh there you go it's empty so that's why it says we don't have a website

[5:39] website so presumably something broke build website oh there you go i don't know that's not good all right no worries i will fix that later you don't have to watch me do that

[6:10] later you don't have to watch me do that what i was trying to tell you is that i've broken up the notebook into four sections um a bit about kind of arrays and numbers and strings a bit about some basic functions some basic operators and then the apl competition stuff because i thought it was getting a little bit unwieldy now there is some helpful stuff about glyphs to cover on the forums

[6:41] to cover on the forums which i think yeah this is probably a good place so thank you roger s for this list and i think somebody else maybe throughout i had one too um and so some of these are pretty easy to get going with so um what i might do is just move this one out of the way so i don't get distracted

[7:11] so some of these are pretty basic functions so i might just add them over here maybe this factorial count as a basic math operation or i wonder what do we have as basic math operators oops yeah i guess it's factorial as well as binomial exactly

[7:42] binomial exactly i'm gonna i'm gonna call it basic okay presumably they call this something like exclamation mark or exclamation point um oh and let's run this line okay uh help

[8:15] exclamation mark okay i guess it doesn't really matter if it's exactly perfect but exclamation okay so the magnetic version is factorial which probably does if anybody not know what factorial is one thing to be aware of is that um [Music] these are always prefix operators

[8:46] so that's going to be confusing um and i guess in operators we could then mention [Music] that factorial is probably going to be the same as times slash iota right so maybe we could use that here um

[9:17] so times oh and we need our apl thingy okay times slash iota five and factorial five there we go look at me i'm doing apl all right

[9:50] um i guess we should probably look it up in case they do weird things with negative numbers or something oh here we go negative numbers so i guess like a factorial of a non-integer is a gamma function is that yes great look at me remembering math from university i've never used that since but i'm sure it comes up for some reason um

[10:28] what's the keyboard shortcut to insert the lamp the comment the comment character uh column uh comma come on i mean you're not working i guess you're a bookmark but yeah i thought it'd be a nightboard or for me but

[11:07] why does it add a dot sometimes it's weird sometimes when i put a couple of spaces it seems good it does all right um and don't ask me to explain gamma because i damned if i remember what it is maybe if there's interest at some point we should come back to that and i guess they're probably going to call this binomial or something [Music] yeah yes

[11:38] double space on mac is period oh it's a mac thing is there a way to turn that off not sure that's right i'll look it up later thanks let me know who to blame okay probably worth mentioning how to define

[12:09] probably worth mentioning how to define binomial [Music] so um

[12:43] results that if i derive from the beta function i don't remember that at all that sounds like an interesting thing to really look up one day as well okay so r is the number of selections of x things from y things that's basically the definition of binomial that we learn in university actually high school at least in australia we did it in high school

[13:24] bit over-enthusiastic with the use of non-integer versions i'd be inclined to do something like say iota 5 um oh we haven't done iota yet no we have people we have done with you though not at this point oh at this point at this point

[14:01] there we go so there are five ways of selecting one thing from five things ten ways of selecting two things from five things and so forth and i guess in one way of selecting zero things does that make sense yep all right great next well

[14:33] um how about we do these other boolean ones they're probably going to be nice and easy to zip through as well okay

[15:03] um so boolean ones here we are let's start here shall we greatest common divisor slash or is nine

[15:42] so this is going to be quite fun because i love the way they don't just do or classic generalization trick which is to say well or is just a special case of gcd which is a very nifty insight

[16:13] um now something i don't know is i haven't really thought about it is if there's some [Music] fundamental reason why that is or it's just a coincidence the greatest common divisor is the same as or

[16:44] the greatest common divisor yeah so the greatest common divide is erratic of zero and zero is zero the greatest common divisor of one and zero is one i i thought those were binary numbers of certainty they're also yeah they're also binary numbers this is also a binary truth table so this is true this is true this this this this is true both for gcd and for the or truth table no no but but so the way i'm reading

[17:15] no no but but so the way i'm reading this and maybe this is wrong the result is a number that's greater or than any of the two numbers above right so how can it be the greatest common divisor this is this is not a number space means create a list in apl yes yes yes but uh all right so it still operates on a single item and that's just that i saw zeros and ones i thought oh wow maybe there's a binary number that's not happening got it

[17:46] got it wise one or zero ah okay that's that's interesting that this is i mean that power can be uh yeah that's cool that's cool uh they also have an a similar interesting point about not having a xor exclusive or because that's uh yeah but maybe that will be in a second oh and then you can also operate on

[18:17] oh and then you can also operate on numbers that are not binary numbers that are okay got it sorry yeah my makes a lot of sense absolutely great um so i think i think the uh the the header um you have uh equal under bar slash there oh thank you as well and what is this thing actually called um

[18:48] um i think it's just one of the greatest kind of devices let me lock it up i accidentally started with dyadic is there a magnetic version of it i think so there isn't no not defined they've actually got a spare character they can play with it looks like it's just called logical or i guess

[19:23] that's easy logical or and uh logical and is that same thing no dyadic yep lowest common multiple or and okay and all right

[20:06] so this is lowest common multiple oh website is up again our website is up again that's what uh the chat says that's weird uh well seem looks like maybe he fixed it oh what a champion okay so yeah so we've got multiple things here now as you see

[20:45] thanks russian he's meant to be on vacation in melbourne why is he fixing our website another kind person you know or maybe he's getting in trouble with his girlfriend for doing that instead of checking out the coffee in melbourne all right

[21:19] and um [Music] i'm gonna guess it's eight nope all right zero i think zero okay i've noticed that most of the time um the uh there's something pointing down the wall the key to the right of it is gonna be pointing up yeah it seems to go down and then up a lot of times

[22:01] cool okay next squiggly versions they're not versions okay makes sense ah and these are now the equivalents with uh to order through them of course that makes sense and they also are the equivalent shifted versions

[22:42] fifth nine oh it's not so pretty in this font oh well when we publish it it should look pretty oh this is only nora is it wow come on hope it had some weird fancy way okay

[23:20] truth table for nor oops this one wow so it's so it's actually an upper an operator that does very little just one thing so

[23:51] thing so yeah it's so unusual oh yeah yeah not inspiring i can just imagine as adam is watching this he's madly searching to prove us wrong define generalizations of nor so he can put it in the next version save himself from this kind of embarrassment in the future ah there we go that's a nice talking font

[24:26] okay you don't think nand is going to be just as disappointing do you i think it is oh boy it makes me wonder why we bother well universe is just helping you get through a lot of symbols yeah well quite thank you yeah trying to do a world record here

[24:57] yeah trying to do a world record here okay everybody happy with our those so far and we've got tilde somewhere builder somewhere anybody's well i'll just type it i don't know how to type children oh of course it won't be in the back in the language box i don't need it in the language bar help children

[25:28] help children okay there we go dyadic and magnetic come on attic tilde is not

[26:05] um well presumably two things for that yeah i don't know why they've got four things trying to make it look more fancy than it is i guess okay uh what happens if we do 9 or negative 2.3 it doesn't like it there you go

[26:36] there you go knots a bit of a disappointment as well i'd be inclined to have it work like python where anything that's not zero or an empty string please turn into one okay it does don't worry we're coming without or excluding

[27:16] okay three one four one five without five one looks like it's removing the actual numbers makes sense but stuff that doesn't appear on the right yeah exactly and it's doing them on whole kind of arrays

[27:47] kind of arrays which you can see in this case remembering that a string is just an array of characters so presumably if we did something like one two three without one two interesting oh i probably need to be careful with my parentheses interesting how is that different you have to enclose the one two on the

[28:19] you have to enclose the one two on the right make it a uh one two here's a scalar oh i see what you're saying this is actually not one two at all this is actually just a parenthesised array yes yes i see uh what's the keyboard shortcut for in close again z oh good job isaac thank you thank you thank you and i don't think

[28:49] and i don't think well i don't think i need that parenthesis because the space should be tightly bound yeah yeah yeah okay we're good let's check the docs to see if there's anything i don't think you need to parent this on the on the right either no i guess i don't really do i yeah and if that actually makes it more obvious what's going on yeah thanks okay so it's uh checking for

[29:22] okay so it's uh checking for equal underbar makes sense and they use hash tables okay makes sense too which is basically the same as python

[29:52] great um actually the problem with looking just at symbols is that we miss out on some and did serata put on list i think on the main topic yes she did

[30:24] yes she did okay she's got all these ones this is going to be more complete i think um all right i guess we're out of all ends so maybe we should do this one presumably this is something like the other shoe um i thought we did that last time though did we i thought we did the opposite direction

[30:55] i thought we did the opposite direction we definitely did the left shoe yeah we got left shoe oh no right shoe okay now the shoe's on the other foot also gotta up shoe and down shoe as well oh really yeah okay uh what's the keyboard shortcut um so

[31:26] um so it'll be x i believe so i'm just pasting it at the moment okay help well oh yes apl you believe right isaac

[32:00] okay we have called ml1 so this is disclose or first and diatic is pick okay

[32:30] all right well these examples are nice and straightforward it's oh well interestingly though in the second case it's also um like unboxing it or whatever the word they use is unenclosing it is it

[33:04] why is that i guess that's interesting is that just how it prints them that's how it prints characters yeah yeah sorry what you're going to say [Music] i don't know if it's is it disclosing it or is it i mean if you just take the one two in parentheses then that would not be enclosed so right okay so my question then is i guess um remind me to enclose keys

[33:42] yeah it's definitely doing that okay so it's taking the first thing and unboxing it but i guess disclosing is what they used to mean unboxing let's see how they describe it it's the first item

[34:15] now i guess you um can't exactly i don't i have a lot of trouble understanding enclose to be honest um because like i don't like in j it's easier there's a data type called a box you know and like you use less than put something in a box and now you've taken that thing and now it's a

[34:46] you've taken that thing and now it's a different data type it's the box of that thing it seems in apl that's not quite what's happening it's still this is still an array but it contains an array they have this idea of depth instead um and so i guess when you select something like the first element by definition you you are reducing the depth you know because that element is you're going into it to get it out

[35:16] going into it to get it out um i remember adam said this was brought up also this is not an array an array and he said oh look at this epsilon down below right yeah i found that confusing um he said it packages it but like you never get the package you know it always acts like an array as far as i can tell it makes an atomic thing

[35:49] so it kind of feels like you know it's different to j i think in j if you select the first element or something and it's in the box you get back a box and you have to unbox it whereas apl doesn't behave quite that way do we get how these left shoe and right shirt are the opposite of each other i don't understand this concept of first that doesn't yeah how much we do because this is so this has depth two because enclosed took something [Music]

[36:21] [Music] which was an array and put it inside an array so we now have an array with an array unit so that's depth two so here's the same thing [Music] here's the same thing disclosed which has taken the first element and therefore it's got a depth one less than we used to have and so therefore it's done the yeah it's it's taken us back to to this um

[36:51] um so to put it another way we could say like a is this and then we could say this matches which is colon hey i should do it the other way around a matches yes

[37:24] hmm okay and so that's a bit of a weird one disclose of

[37:55] that's a bit of a weird one disclose of them with the zelda thing is zero in edge case all right so let's look at the direct version shall we pick this looks like it's just indexing in okay so if you've got a scalar on the left it just grabs the third thing or grabs the second thing [Music]

[38:26] [Music] this goes into the second thing and then finds the first thing in the second thing all right seems easy enough [Music] okay well that's

[38:57] well that's doing fine if there's anything weird about this to know about okay elements of x selection successively deeper levels yep simple scalar items may be picked by empty vector items arbitrary depth what does that mean [Music]

[39:35] okay what's going on with these ones but they've got two sets going on [Music] so this is um that was row two column one is jkl i'm getting so confused as to which switch okay we're talking about this one at the moment so this one here is enclosed

[40:06] enclosed so that creates an array with an array in it and then it concatenates one wait isn't that exactly the same as just putting a space between them yeah why are they confusing us okay um and then oh

[40:37] and then oh that's interesting so now they're using that as an index and why are they building up g in such a weird way um all right let's i think we are going to need this example because this is something i

[41:11] i think with the boxing this would be uh made pretty clear what's going on yeah okay and not having the boxing there i think is making it look a lot harder than it really should be really okay maybe they were running out of room or something i don't see why they don't just do this

[41:50] okay so we've got this weird thing so i think we index in second row first column gives jkl4 first element of jkl4 gives you jkl okay so all this is the index and we're sorry what's the keyboard shortcut for

[42:20] keyboard shortcut for x and we need to fix this x okay so sorry we we go into the um second this has been going to the second row first column i believe so yeah and then this is crap the first thing

[42:53] and then this is crap the first thing yeah from that um well we should try that on a matrix to confirm so um two rows three columns of iota six okay so you can't grab a row from a matrix

[43:27] um okay so oh and you need the left true because you want that to be an atomic thing the index i think so

[43:58] i think so yeah it's gonna give you an error um what's this cable shortcut for quad l okay so that's the second row first column um

[44:29] and if you want to entire row do you say the entire cell um well we could try doing this i guess no so i'm not sure you can we just did it with g didn't we no no we didn't no so the difference is um okay

[45:00] didn't no so the difference is um okay so it looks like you use um a plain array to [Music] go into [Music] an array of arrays as in this one so this is this is an array of arrays to go into a

[45:31] a matrix you need a s rank one or shape one thing i guess such as we have here and so here now we've got a shape one thing [Music] and then a scalar so the shape one thing is going to go into the matrix to row two column one

[46:01] one and this now contains two items so this is the first item okay so i think it's one way to get the row is to use the up arrow and down arrow yeah exactly okay that's weird um

[46:33] so the items of x are simple integer scalars or vectors okay yeah so when we enclose them we're creating a vector and the vector identifies a set of indices one per axis at that level of nesting okay so that description makes sense after i know what it means i'm not sure i could have read it to figure out what it means i still don't know what simple scalar items in y may be picked by empty vector items in x

[47:06] an empty vector how do you even create an empty vector um is that iota 0 no i don't know i might leave that one for now because it's not going to help zip through it something to maybe somebody can try to figure that out okay that was annoyingly long for our hope but such as life um [Music]

[47:36] [Music] should we do the up tech and down tech if that's what they're called yes is that what they're called does anybody know uh yes that's what they're called okay and how do i type them um and then as in nancy oh

[48:08] so the the one with the pointing upward and b and then is kind of goes down oh um d oh nancy and barry so this one episode tech

[48:43] okay and that's only diet decode okay this should be fun

[49:14] oh nurse is very confusing the first element of x has no effect on the result this function's only the base one that's

[49:47] okay here's a polynomial it evaluates so we start with y1 okay this is y1 times x to the n minus 1 times 2 to the zero so 2 plus 4 plus oh sorry uh x to the n minus one x

[50:20] x to the n minus one x x is on the left oh sorry and that's zero i guess yeah that's zero so okay so it's one times one plus one times two plus zero plus one times that's the usual formula that we use to convert binary to decimal yeah i understand i'm just trying to see how it maps to that oh yeah okay so this is a binary this is the binary number 1101

[50:50] binary number 1101 because we're doing it base two okay so then um [Music] what about base ten yep makes sense what you're you're telling it what is one one zero one okay so that's one times ten to the zero

[51:21] okay so that's one times ten to the zero plus one times ten to the one blah blah blah so we should be able to do three four one six and that should just be that number right yep oh all right um when you've got an array on the left each element of x defines the ratio

[51:51] each element of x defines the ratio between the units for the corresponding pairs okay so the ratio okay so the ratio of the units is 24 to 60 to start with so presumably this is like presumably what this is is um two hours the two days 40 sorry 2 hours 46 minutes

[52:23] sorry 2 hours 46 minutes and 40 seconds so that would be two hours i've got to do parentheses and [Music] 46 minutes and 40 seconds okay uh why

[52:54] so i see so we start out with 40 and so this unit's considered one

[53:27] and so then the next one is going to be 60 times okay so here's the 60 times the next one will be another 60 times against i should write this as 60 times 60 to be more clear um yes and i must have pressed insert i see and so the 24 indeed is not doing anything

[54:02] okay that's fun um so that's uptech higher rank array arguments whoa okay each of the vectors taken as the radix vector for each of

[54:32] taken as the radix vector for each of the vectors on the first axis first axis is the rows jeremy before you move on do you mind to uh the last example or the polynomial or evaluation is a that is a compass number do you want to put that in it as well oh you're right sure um so that's going to be

[55:07] um [Music] one j one to the power of zero plus one j one to the power of 1 times 2 is that how it works yes plus 1 j 1 to tau 3 right to the power of two times three

[55:40] um so one j one to the power of zero is obviously going to be one and then 1 j 1 to the power of 1 is obviously going to be 1 j 1 and that one's times 2. okay and then we've got 1 j 1 squared which is going to be [Music]

[56:12] [Music] 1 one j one squared um [Music] which is one plus i times one plus i which equals 1 plus 2i plus minus 1. okay which is indeed 0j2 okay let's give us another of those and then

[56:42] let's give us another of those and then that times itself again um [Music] no screw it's easier to see it in the polar coordinate why is that oh because 45 degrees and 90 degrees then okay sorry i'm not much of a complex numbers person so all right so j1 is

[57:12] numbers person so all right so j1 is the one comma one on the cartesian plane is that what you're saying right one j1 angle phase is 45 degrees and when you do times is that the same as multiplying right power goes translates to multiplying by phase and also magnitude gets to that power okay i maybe you can find some link or something that explains that for dummies like me and put it in the forum or

[57:43] like me and put it in the forum or something for this lesson because i never learned any of that when i was studying philosophy at university i'm afraid um okay um so this one is four by one two three four five six seven eight and here's the eight

[58:15] and here's the eight um x is the thing on the left i still wish they said alpha instead of x so each of the vectors each of these vectors is taken as the radix vector for each of these vectors uh but there's three of these rows and

[58:45] uh but there's three of these rows and four of these rows how does that work oh i see it's doing an outer product so it's doing it's doing this with this and that's what this is um is that right

[59:18] that's now i think we need to explain in terms of radix i don't know what a radix vector is it's like the base um like 13 in radix 2 is 1 1 0 1 just like above um so no okay that's i'm wrong about how these are combined

[59:55] so how these how do these work i don't get it either i think apl wiki has a better explanation oh great as always but still complicated app tag

[1:00:25] search for decode team card so examples the third paragraph under example is the third partner one two three yep okay so we've got these ones

[1:00:59] okay this is the polynomial okay okay we've done this one one tack one tech is um oh okay so that's um

[1:01:30] i keep forgetting how this works x it's on the left and then we multiply it by oh that is the sum right 1 2 power of anything is one so one pair of anything is one yep um

[1:02:02] so we end up with three times one plus one times one plus four times one etcetera that makes sense it gets tricky when the left hand side is an array yes okay so then um doing it here with a matrix

[1:02:33] is um [Music] oh where interesting is this sum the column yeah why like what's it doing is it doing it to each like yeah i guess that's my question why why

[1:03:14] um oh if the left argument is a scalar it's converted to a vector filled with that number

[1:03:47] ah oh so it's one one vector which size columns find out that one um okay so it's basically aligning so okay so because d code expects kind of a a scalar on the left and a vector on the right

[1:04:24] it's not quite sure um [Music] oh i don't know what 48 did so um the kind of the base case seems to be more this version and then the one where you've got a scalar is just a kind of a shortcut all right

[1:04:57] the scalar one is good for converting to bases yes or deconvert into bases yes go to base 10. 31 is useful come to for converting to days and minutes and the one that the left-hand side is an

[1:05:28] the one that the left-hand side is an array it's not clear what the use is uh the one with the left-hand side is array i mean it's useful for like the kind of example they gave here i guess so how many seconds is 2 hours 46 minutes and 40 seconds right yeah i meant when the right hand side is a matrix and left hand side is that right okay right the one there what if we do two two two what would that do just a moment here uh there's 12 of these

[1:06:05] okay so it doesn't like that so it has to be three okay two [Music] is that no oh yeah 23 is one plus

[1:06:38] uh 5 times 2 to the power of 2 now to the power of one plus nine times four is it no it's not um [Music] it's one to the power of zero times two

[1:07:09] one to the power of zero times two plus five to the power of one yeah but that's not it doesn't even say what it does in that case does it um the first one is ignored then oh this is the ratio of them um [Music] so [Music]

[1:07:40] with ratio two to two should be similar to a ratio of one so it doesn't really do anything all right well maybe we'll come back to this um we're slightly lost on that one yeah i was looking at this before the

[1:08:11] yeah i was looking at this before the class also and couldn't figure it out all right um yeah it's something going on interesting here as well all right

[1:08:41] um [Music] i hope i'm not expecting but i'm hoping downtap is less confusing

[1:09:13] oh and what's this called uh decode presumably this is going to be something called encode yes [Music] all right it's gonna be the opposite

[1:10:17] um [Music] okay what's going on here that's a hard one to start with

[1:11:01] okay so this is decoding seven to a binary number of length according to whatever's on the left oh do left hand side columns but here's five

[1:11:33] but here's five here's seven and here's twelve that's right okay so we should be able to do the same thing

[1:12:21] oh yep y and x must both be simple numeric arrays it's the representation of y in the number system defined by x if the first element of x is zero the value will be fully represented

[1:13:00] so you're getting the digits of the right hand side in the bases specified by the left hand side one by one oh there you go that's the difference so if the first element's zero then it's going to make this as big as it has to be to get the whole number here if it's not zero then it truncates it

[1:13:31] makes sense well truncate is this very special case where the thing is 10 not really um if the thing was binary this is a truncated binary representation it's the least significant digits in that representation right here okay yeah you're getting the digits one by one in that basis

[1:14:05] this is how you get the least significant digits okay let's see what you mean by truncated now okay

[1:14:38] so here's 75 in base 2 is probably column 1. i don't understand column 2. yeah column two is up tall oh okay so that's representation of 75 in base heat yeah and the other one is hex right hexadecimal except we don't have letters

[1:15:12] oh so it's b4 up tool three one one binary 1101010 it's hard to imagine that one would want to do this i mean it could be some kind of performance like things sometimes [Music]

[1:15:42] [Music] where you need yeah multiple representations at once but i don't think we need it for a quick like what are these glyphs yeah all right make sense wait we've done these yeah which one those napoleons we've done that all right so let's do

[1:16:14] all right so let's do up and down shoe might be easier ones to work oh these ones here yeah do they make sense to go with left shoe and right shoe isaac oh not really it's unique intersection and union that's what they are oh okay all right let's go they don't really match with those

[1:16:45] um duplicate the rest of the glyphs

[1:17:36] uh rest i guess we could give it a number rename okay

[1:18:09] all right how do i write these they're going to be c and v and this is going to be called appshoe oh yeah and what are their names of the things they do uh up shoe is just um intersection it's just diatic it's any dyadic yep

[1:18:41] yep sorry and then it's called set intersection more precisely oh i see diet set intersection means intersection yeah weird um yeah sorry you're saying

[1:19:11] yeah sorry you're saying okay this one's v right yep um that one the magnetic is unique and uh dyadic is union attic nick dyadic union

[1:19:48] okay that looks straightforward enough so intersection of 22 appears here a b appears here f g does not appear here and vice versa okay this is similar to something we saw before right um

[1:20:19] something we saw before right um the tilde i think um but it would be you have to use the tilde going both directions you'd have to use it twice i think to get the intersected right can we try to do that then um uh so if we wanted the intersection of

[1:20:57] um yeah so this is wrong isn't it it should be like this answer to this should be one so in venn diagram

[1:21:27] so in venn diagram tailgate stuff that it's outside the difference yeah so it's um the set minus this and minus this okay yeah unionized too yes exactly okay

[1:21:58] okay um [Music] great anything weird about this these actually match their math notations for once yes all right that's nice and simple okay

[1:22:49] hey so the diet version seems pretty simple the monadic also looks pretty simple and we've learnt a way to do this before

[1:23:23] and we've learnt a way to do this before um [Music] using not equal yes exactly and re reduce was it slash not equal yes that sounds right i mean logical it's like a four or a six or something

[1:23:53] was it that only the one with the three things that one right which is back to the index into index intel yeah um i think you are right actually it's that's the one [Music]

[1:24:26] yes and um i think we looked at doing this um

[1:24:58] let's put it another way it's this um and then did we come up with a way to do that with a with a train you should be able to whenever a is on both sides yeah i mean it would draw it's an ugly version of it i think

[1:25:33] no why is that wrong i thought this means put it on both sides that does but uh the the tilter in between shouldn't that be a chart this one uh-huh oh there does have to be a jut there yes i think so that looks familiar but um yes yes uh what is jobs how do you type j j there we go

[1:26:04] there we go um oh do you need that tilde inside parentheses i think so um let's see we applied this to the a and then yeah because we have to reverse it this actually has to go on the right hand side because it's here if it's on the sorry if it's on the left hand side if it's on the right hand side we find but we need to not equals to apply to the left hand side oh so we have to reverse the order of slash

[1:26:37] so yeah interesting question as to how to do this more neatly i'm sure there's a way okay that's that

[1:27:11] um okay let's leave it there i'm gonna do an extra one tomorrow if people are around um [Music] in the hope that we can [Music] get close to polishing this off so this is like learning chinese isn't it grooves well there's less to learn than chinese

[1:27:45] thankfully they're more complicated to understand what they mean than chinese glyphs though you would know better all right thanks all right thank you bye-bye
