# Neural Style Transfer in Tensorflow - ShortNote

This personal note is perhaps for a person who is already familiar with the concept of neural style transfer.
For peoeple who are new in this area, it is recommended to read reference's section belows:

## What is Neural Style Transfer
- Neural Style Transfer is the technique that creates a new image with a certain style of artistic image
- It can be understood easily through the examples below:
<image>
  
## High-Level Intuition
- when the input image passes through feed-forward convolutional neural network, each layers act as a collection of filters, which extract certain features from the image
- It is found that, with pre-trained weights, style and content can be obtained
- Therefore, it is possible to merge both of them together to create a new image 
- <image>

## How to implement

### Pre-Processing
- Firstly, we need to build the VGG model
- Then, load pre-defined weight
- This part, with some modification, I use code from *Chip Huyen, CS20: "TensorFlow for Deep Learning Research", Stanford*

### Model Implementation
- Unlike traditional nerual network, we train the model to update the generated image
- In other words, the weights and biases of the model are fixed but the input image (generated image) is trainable

#### Loss function
- In neural style transfer, loss function can be broken into Content Loss and Style Loss
- ![test](./Images/loss_image.jpg?raw=true "Title")

##### Content Loss
- To calculate content loss, we use the square error to calculate loss based on two values
  - activated function's value of content image in a content layer
  - activated function's value of generated image in a content layer
- Particularly, this solution uses *CONV4_2* as a representation layer of content
<image>

##### Style Loss
- Calculating style loss is a bit trickier
- First, it is needed to calculate the Gram Matrix 
  - In a nutshell, Gram Matrix is used for finding the correlations among filters on a certain layer
  - <image>
- After getting a Gram Matrix of both style image and generated image, we calculate the square error to obtain style loss
- <image>

## Dependencies
- Tensorflow
- cv2
- Numpy
- urllib

## References
- [Neural Algorithm of Artistic Style - Leon A. Gatys, Alexander S. Ecker, Matthias Bethge](https://arxiv.org/abs/1508.06576)
- [CS20: "TensorFlow for Deep Learning Research", Stanford](http://web.stanford.edu/class/cs20si/)
- [DeepLearning.ai](https://www.deeplearning.ai/)
- [Neural Artistic Style Transfer: A Comprehensive Look, Shubhang Desai](https://medium.com/artists-and-machine-intelligence/neural-artistic-style-transfer-a-comprehensive-look-f54d8649c199)
