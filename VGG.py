
# coding: utf-8

# In[1]:



""" Load VGGNet weights needed for the implementation in TensorFlow
of the paper A Neural Algorithm of Artistic Style (Gatys et al., 2016) 

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

For more details, please read the assignment handout:
https://docs.google.com/document/d/1FpueD-3mScnD0SJQDtwmOb1FrSwo1NGowkXzMwPoLH4/edit?usp=sharing

"""

#ipython notebook --script
import numpy as np
import scipy.io
import tensorflow as tf

import utils


# In[13]:


# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

class VGG(object):
    def __init__(self, input_img):
        
        ### download file from the url
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME)
        
        ### get data from downloaded file
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.input_img = input_img
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
        self.dataDict = {}

    def _weights(self, layer_idx, expected_layer_name):
        
        ### Return the weights and biases at layer_idx already trained by VGG
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):

        with tf.variable_scope(layer_name) as scope:
            weights, biases = self._weights(layer_idx, layer_name)
            weights = tf.constant(weights, name='weights')
            biases = tf.constant(biases, name='biases')
            conv2d = tf.nn.conv2d(prev_layer, 
                                filter=weights, 
                                strides=[1, 1, 1, 1], 
                                padding='SAME')
            out = tf.nn.relu(conv2d + biases)
        self.dataDict[layer_name] = out
        

    def avgpool(self, prev_layer, layer_name):

        with tf.variable_scope(layer_name) as scope:
            out = tf.layers.average_pooling2d(prev_layer,
                                             pool_size=2,
                                             strides=(2,2),
                                             padding='SAME',
                                             name=layer_name)
        self.dataDict[layer_name] = out

    def load(self):
        self.conv2d_relu(self.input_img, 0, 'conv1_1')
        self.conv2d_relu(self.dataDict['conv1_1'], 2, 'conv1_2')
        self.avgpool(self.dataDict['conv1_2'], 'avgpool1')
        self.conv2d_relu(self.dataDict['avgpool1'], 5, 'conv2_1')
        self.conv2d_relu(self.dataDict['conv2_1'], 7, 'conv2_2')
        self.avgpool(self.dataDict['conv2_2'], 'avgpool2')
        self.conv2d_relu(self.dataDict['avgpool2'], 10, 'conv3_1')
        self.conv2d_relu(self.dataDict['conv3_1'], 12, 'conv3_2')
        self.conv2d_relu(self.dataDict['conv3_2'], 14, 'conv3_3')
        self.conv2d_relu(self.dataDict['conv3_3'], 16, 'conv3_4')
        self.avgpool(self.dataDict['conv3_4'], 'avgpool3')
        self.conv2d_relu(self.dataDict['avgpool3'], 19, 'conv4_1')
        self.conv2d_relu(self.dataDict['conv4_1'], 21, 'conv4_2')
        self.conv2d_relu(self.dataDict['conv4_2'], 23, 'conv4_3')
        self.conv2d_relu(self.dataDict['conv4_3'], 25, 'conv4_4')
        self.avgpool(self.dataDict['conv4_4'], 'avgpool4')
        self.conv2d_relu(self.dataDict['avgpool4'], 28, 'conv5_1')
        self.conv2d_relu(self.dataDict['conv5_1'], 30, 'conv5_2')
        self.conv2d_relu(self.dataDict['conv5_2'], 32, 'conv5_3')
        self.conv2d_relu(self.dataDict['conv5_3'], 34, 'conv5_4')
        self.avgpool(self.dataDict['conv5_4'], 'avgpool5')


