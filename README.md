# Junwen-Real-Time-Gender-and-Age-Detection-with-OpenCV

# Objective:

To build a gender and age detector that can guess the gender and age of the person (face) in a picture or through webcam.

# About the Project :

In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

# Additional Python Libraries Required :
<li>OpenCV</li>
<li>argparse</li>

# The CNN Architecture
The convolutional neural network for this python project has 3 convolutional layers:

<li>Convolutional layer; 96 nodes, kernel size 7</li>
<li>Convolutional layer; 256 nodes, kernel size 5</li>
<li>Convolutional layer; 384 nodes, kernel size 3</li>

It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

<li>Detect faces</li>
<li>Classify into Male/Female</li>
<li>Classify into one of the 8 age ranges</li>
<li>Put the results on the image and display it</li>

# Demo video of application

<video width="640" height="360" controls>
  <source src="Demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
# Junwen Wang - Virginia Tech
