## Introduction
A pistache is an artistic work that imitates the style of another one.

This culmination of half a semester of work for my independent study has led to the current state of an edited android app, in which I implemented image-segmented background blurring, and increased luminance matching to a Google Codelab [1], that focuses on creating pistaches.

After getting to this point, and remembering that this independent study was supposed to be about machine learning and art, and I hadn’t made any art projects, I began to think about what I wanted to make with this unique application.

I quickly realized that my moms birthday was coming up, and that I have historically never gotten her anything. For those who don’t know my mom, I personally believe that she has a manic addiction to her children. That belief is entirely centered around the life she wishes to portray through her Facebook. 

Facebook photos tell your story, either intentionally or unintentionally, to everyone that kinda knows you.

Either way, in deciding how to celebrate the birth of my mom, I came to the conclusion that I wanted to tell her story. The one she tells through Facebook that is. It isn’t necessarily the story I would have chosen to portray, but it’s the one she has.

So, I had the pleasure of scouring Janice’s photos to build my collage, and I grew increasingly fearful for my privacy in the process. Regardless, I grabbed about 60 photos from her facebook (and some others I wanted to throw in), and created 26 different pistaches with styles ranging from Picasso to Van Gogh, resulting in over 1500 total photos. From the 26 stylized photos, I then chose the one that I thought most truly represented that memory to me. Thus, trying to portray how I view the world that she is portraying. Seeing her world through my eyes.

Also, for fun, I took all 1500 photos and put it in a super-collage, I call the meg. That photo is 170MB.
General Overview of Semester
Moving on, I had went for a run one morning, with the task of figuring out what I wanted to do for an independent study this semester. The run was through my campus, and then into a wooded park near my university. About two minutes into the walk, my mind had already wandered from my original task, and I started to think about something cool that my professor had sent me, earlier in the summer, [2]. It was a 360 degree virtual reality experience of Van Gogh’s Starry Night world. When I watched it for the first time, I freaked out, and thought it was the coolest thing ever made. 

Anyways, that made me think about doing something like that. I really liked what that was, but it was restricted to that piece, and I wanted to see the world like that. I wanted to see the world through the lense of a famous artist, or any artist for that case. Which is what led me to read up about the style transfer, and the possibilities of experiencing the world in real time on a mobile device.

After formalizing, my independent study, I did basic research as partially described below, where I read most of the more distinguished papers on the topic. The three papers I discuss, led to the final creation of a Google Codelab [1], which in fact already implemented pretty much exactly what I wanted to do. Both a bummer, and a blessing. Using the app already created by Google, allowed me to focus on different aspects of things I was interested in. I implemented the soon to mentioned Increased Luminance Matching, as well as for as far as I can tell, was the first to artistically use image segmentation with style transfer for the purposes of Lens Blur, where the non-focus of the images are significantly blurred to create portrait-esq photos. This moved away from my initial ambition of creating an application that allowed users to become completely immersed in the environment, but I deemed it a positive step in creating something unique.

It wasn’t a direct path to the end result that I currently have. As I am doing an independent study, and because this isn’t a part of any course, there isn’t really any guide for how to learn or do something. You kind of just have to figure it out. There were many failed attempts, and a lot of time went into finding the right things to use. When you blindly walking through a forest, and you don’t necessarily know what you are doing, it’s hard to find the path, and in many situations I threw hours and hours into dead-end after dead-end, and even though they didn’t lead to any tangible results, the processing of failing epically made me constantly understand more and more the problem I was trying to solve. 

The problems that plagued my experiences revolved around getting a working model from one that runs inference on my computer, to one that works for Android. There are a number of tutorials for this, but in every situation, something was always pre-supposed. And in every situation, the things I was working with didn’t meet the criteria. Ultimately, I found exactly what I was looking for, but I think the process of struggling to find it has led me to have an infinitely better understand of what I am using and why I am using it. 

For reference the model I found that ultimately worked was [3] Google Deeplab. This has modified versions of the model that are specifically made for mobile devices. For those interested, the general problem with other models are that they might use certain layers that aren’t compatible with Android devices, or they might use certain features that aren’t initially loaded as part of the TensorFlow library for Android.

## Process
Applying Neural Style Transfer to images results in beautiful images that really feel like the artist themselves hand generated the image. But, in many cases the resulting images loose the real-world nature of the image. In this report, I will outline a few methods in which I aimed at bringing back the ‘realness’ of an image.

There are two primary contributions that I have added into artistic style transfer to make resulting images more realistic in post-processing, Increased Luminance Matching, and Segmented Lens Blur. 
Increased Luminance Matching
In many situations, resulting images can get completely absorbed by the style they are trying to replicate. One way to break this is to simply weigh the style less and less, which leaves an unsatisfying resulting as the original image intrudes on the new image being created. To counteract this, I have implemented luminance matching. Luminance matching kind of already done when the image goes through the network, but I have found that increasing the intensity in post-processing can correlate with more realistic photos that still maintain the properties of the stylized image. Luminance matching is performed on a pixel-by-pixel basis, in which the pixel of the stylized photo is increased or decreased depending on its luminosity in proportion to the corresponding pixel luminosity of the original photo. 
Segmented Lens Blur
Using a pre-existing model for image segmentation[], I was able to successfully implement it on an Android device with the help of []. After gathering the segmentation within an image, where my images usually focused on photos of people, I simply masked the segmented people of the one image onto another. The resulting images has mixed results, depending on a number of things. Mostly, if the stylized image had pre-defined lines and edges around faces and bodies, then the resulting images looked very legitimate.

## Mid Semester Summary
Additionally, there are numerous other things that I have learned and failed through so far. I went into this semester with barely any knowledge of anything in this realm, and I really feel like I have learned a lot up to now, and am excited to continue to learn more and more.

The first few weeks of my semester were spent trying to learn and understand the mathematics and qualities that modern day style transfer is based off of. I read a few of the dominant papers in the field that discussed the methods. 

To summarize, when creating a stylized image, there are three main players. The content photo (c), the style photo (s), and the new stylized photo (n). 

At a high level, we want to make n as similar in content to c as possible, and as similar in style to s. So how can we define, what ‘content’ and ‘style’ are? They found that Convolutional Neural Networks (CNN’s), when trained for object recognition, extract different types of information at different layers. If you then try to minimize the feature reconstruction for particular layers, you can extract different information. Importantly, minimizing early layers in the CNN seems to capture the texture and color images, whereas, minimizing higher layers in the CNN seem to capture image content and overall spatial structure [Johnson].

There is a lot of math that is involved in reconstructing those layers, and minimizing the differences between images, that I won’t get into here. After the initial paper was published, an important addition was made that enabled the created of the Android application that I ended up editing. [Johnson], demonstrated a way of not only creating stylized images, but also creating them in real time. 

To create the stylized image in the original paper, there was both a forward and backward pass through a pretrained network. To fix the problem of speed, they trained a different neural network to quickly approximate solutions to their problem [Johnson]. Again, the mathematics behind this task are complex enough to leave out of this overview.



[1] https://codelabs.developers.google.com/codelabs/tensorflow-style-transfer-android/index.html?index=..%2F..%2Fio2017#0

[2] 
https://www.facebook.com/Artsper/videos/1757088830996244/

[3] 
https://github.com/tensorflow/models/tree/master/research/deeplab
https://github.com/tensorflow/models/issues/4278
