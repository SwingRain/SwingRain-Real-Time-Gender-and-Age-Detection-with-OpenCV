# Junwen-Real-Time-Gender-and-Age-Detection-with-OpenCV

# Objective:

To build a gender and age detector that can guess the gender and age of the person (face) in a picture or through webcam.

# About the Project :

In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

# Additional Python Libraries Required :
* OpenCV
* argparse

# The CNN Architecture
The convolutional neural network for this python project has 3 convolutional layers:

Convolutional layer; 96 nodes, kernel size 7
Convolutional layer; 256 nodes, kernel size 5
Convolutional layer; 384 nodes, kernel size 3
It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it

# Demo video of application


# Junwen Wang - Virginia Tech
