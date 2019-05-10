# Dog/cat Classification
End to end Guide for Image classification by going through the Different Dogs and cats images 

## I am going to share the end-to-end process of building dog/cat classifier using TensorFlow.

When any one start learning deep learning with neural network, a time come when you will realize that the most powerful supervised deep learning techniques is the Convolutional Neural Networks.





## Steps towards making the first image recognition model

1. Collecting the Dataset
2. Importing Libraries and Splitting the Dataset
3. Building the CNN
4. Full Connection
5. Data Augmentation
6. Training our Network
7. Testing

1. Collecting Dataset
We need to have huge amount of data for good training so that model can learn how to recognize unseen data.

3. Buliding the CNN
This is most important step for our network. It consists of three parts -

Flattening

CONVOLUTION
The primary purpose of Convolution is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data, as every image can be considered as matrix of pixel.After convolution We apply RelU function (every time).

Activation function:
It as just a piece of thing to get output from a node.Activation function can be linear or non linear. It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc depending upon the function.
In case of linear output of the functions will not be confined between any range and so it is not usefull for the usual complex data available.We have various non-linear activation functions like RelU , tanh , sigmoid.

### Sigmoid Function 

Its is S shaped curve and its value lies between [0,1] , so it can be useful when we want probablities as output as probabilities also lies in [0,1]. The softmax function is a more generalized logistic activation function which is used for multiclass classification.

### Tanh or hyperbolic tangent Activation Function

Tanh is also like logistic sigmoid but better. The range of the tanh function is from [-1,1].It is also S shaped.It is mainly used in two class classification problem.

### RelU 

The ReLU is the most used activation function.Since, it is used in almost all the convolutional neural networks or deep learning.
f(x) = 0 x < 0
f(x) = x  x >= 0

so its is half rectifier function.But as it makes negative values to 0 so it decreses the ability of model to train data.So there is advanced Leaky RelU which increases the range of the ReLU function. 
f(x) = ax  x < 0
f(x) = x  x > 0
a is 0.01 or so.       ### range of the Leaky ReLU is [-infinity,infinity]
When a is not 0.01 then it is called Randomized ReLU.

POOLING

It can be called down sampling step where we use filter of size smaller then the rectified convoluted image . Generally we use max Pooling it takes the maximum value in the filter and preserve most important features.For example we define a spatial neighborhood 
(2×2 window) and take the largest element from the rectified convoluted image also called as feature map within that window. Instead of taking the largest element we could also take the average (Average Pooling) or sum of all elements in that window, but max pooling proved to give good results.

FLATTENING

After pooling comes flattening where matrix is converted into a linear array so as to input it into the nodes of our neural network.

FULL CONNECTION

Full connection is connecting our convolutional network to a neural network and then compiling our network.

DATA AUGMENTATION

While training we need lots of data if data is not sufficient the model will not get trained to give good results for unseen data
Its not necessary to go and hunt for more data if time donot permit you there is method called data augmentation which will solve this problem.It creates a roted version of image and increse the images. Neural network treats all these images as seperate image.
So, to get more data, we just need to make minor alterations to our existing dataset. Minor changes such as flips or translations or rotations. Our neural network would think these are distinct images anyway.

