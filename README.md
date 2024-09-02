# Junwen-Real-Time-Gender-and-Age-Detection-with-OpenCV

# Objective:

To build a gender and age detector that can guess the gender and age of the person (face) in a picture or through webcam.

# About the Project :

In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person from a single image of a face. I used the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html" rel="nofollow" onclick="javascript:window.open('https://talhassner.github.io/home/projects/Adience/Adience-data.html'); return false;">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

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

# Dataset

The dataset has been linked in the main.py program......when u run the program, dataset will be downloaded automatically and load to the program training set. If u want use a different dataset. You can through the below link from Kaggle datasets.

For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it <em><strong><a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification" onclick="javascript:window.open('https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification'); return false;">here</a></strong></em>.. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

# Steps

For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

We use the argparse library to create an argument parser so we can get the image argument from the command prompt. We make it parse the argument holding the path to the image to classify gender and age for.

For face, age, and gender, initialize protocol buffer and model.

Initialize the mean values for the model and the lists of age ranges and genders to classify from.

Now, use the readNet() method to load the networks. The first parameter holds trained weights and the second carries network configuration.

Let’s capture video stream in case you’d like to classify on a webcam’s stream. Set padding to 20.

Now until any key is pressed, we read the stream and store the content into the names hasFrame and frame. If it isn’t a video, it must wait, and so we call up waitKey() from cv2, then break.

Let’s make a call to the highlightFace() function with the faceNet and frame parameters, and what this returns, we will store in the names resultImg and faceBoxes. And if we got 0 faceBoxes, it means there was no face to detect. Here, net is faceNet- this model is the DNN Face Detector and holds only about 2.7MB on disk.

Create a shallow copy of frame and get its height and width.
Create a blob from the shallow copy.
Set the input and make a forward pass to the network.
faceBoxes is an empty list now. for each value in 0 to 127, define the confidence (between 0 and 1). Wherever we find the confidence greater than the confidence threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates and append a list of those to faceBoxes.
Then, we put up rectangles on the image for each such list of coordinates and return two things: the shallow copy and the list of faceBoxes.
But if there are indeed faceBoxes, for each of those, we define the face, create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values.

We feed the input and give the network a forward pass to get the confidence of the two class. Whichever is higher, that is the gender of the person in the picture.

Then, we do the same thing for age.

We’ll add the gender and age texts to the resulting image and display it with imshow().
# Demo video of application

<video src="https://github.com/user-attachments/assets/4cea7490-f6a3-4bf0-b748-4e68feedfb25" width="352" height="720"></video>

<video src="https://github.com/user-attachments/assets/40e79a40-1240-4e18-859c-30577eb86e76" width="352" height="720"></video>

# Junwen Wang - Virginia Tech
