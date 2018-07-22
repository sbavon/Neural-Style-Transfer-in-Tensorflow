
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time
import numpy as np
import tensorflow as tf
from VGG import VGG
import cv2
import matplotlib.pyplot as plt
import utils


# In[2]:


##############################
### setup hyper parameter
##############################
STYLE_LOSS_WEIGHT = 1000
CONTENT_LOSS_WEIGHT = 1
LEARNING_RATE = 10
CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
### with lower value, the weight on the early layer will be higher
STYLE_WEIGHT_TREND = 0.2
TRAINING_EPOCH = 2000
tf.reset_default_graph()


# In[ ]:



class Style_Transfer(object):
    
    def __init__(self, 
                 content_img_addr, 
                 style_img_addr, 
                 img_width = 224, 
                 img_height = 224):
        
        self.img_width = img_width
        self.img_height = img_height
        
        ### get content image, style image and initiate generated image
        self.content_img = utils.get_image(content_img_addr, img_width, img_height)
        self.style_img = utils.get_image(style_img_addr, img_width, img_height)
        self.gen_img = utils.generate_noise_image(self.content_img, img_width, img_height, noise_ratio=0.6)
        
        ### identify weight used for calculating total loss
        self.style_loss_weight = STYLE_LOSS_WEIGHT
        self.style_weight_trend = STYLE_WEIGHT_TREND
        self.content_loss_weight = CONTENT_LOSS_WEIGHT
        
        ### identify layers that represent style and content
        self.content_layers = CONTENT_LAYERS
        self.style_layers = STYLE_LAYERS
        
        ### identify image inputs
        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE) as scope:
            self.img_input = tf.get_variable(name='img_input',
                                            shape=([1,self.img_width,self.img_height,3]),
                                            dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
            
        ### setup learning parameters
        self.learning_rate = LEARNING_RATE
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.training_epoch = TRAINING_EPOCH

    def load_VGG(self):
        
        self.vgg = VGG(self.img_input)
        self.vgg.load()
        self.content_img -= self.vgg.mean_pixels
        self.style_img -= self.vgg.mean_pixels

    def calculate_content_loss(self, gen_layers, content_layers):
        length = len(content_layers)
        content_loss = 0
        for i in range(length):
            content_loss += tf.reduce_sum(tf.square(gen_layers[i] - 
                                        content_layers[i]))/(4*content_layers[i].size)
        self.content_loss = content_loss
        
    def calculate_gram_matrix(self, matrix, n, m):
        f = tf.reshape(matrix, (n,m))
        return tf.matmul(tf.transpose(f),f)
        
    def calculate_style_loss(self, gen_layers, style_layers):
        
        length = len(gen_layers)
        style_loss = 0
        style_weight = 1
        for i in range(length):
            
            ### calculate n by multiplying width and height of that layer 
            n = style_layers[i].shape[1] * style_layers[i].shape[2]
            ### m is the layer's depth 
            m = style_layers[i].shape[3]
            
            ### calculate style loss for each layer
            factor = 1/(2*n*m)**2
            l = factor*(self.calculate_gram_matrix(gen_layers[i], n, m) - 
                         self.calculate_gram_matrix(style_layers[i], n, m))**2
            l = tf.reduce_sum(l)
            ### accumulate style loss from each layer
            ### Note. multiply style_weight_exp
            style_loss += l*style_weight
            style_weight *= self.style_weight_trend
        
        self.style_loss = style_loss
        
    def calculate_loss(self):
        
        with tf.Session() as sess:
            ### get the content layers of the content image
            sess.run(self.img_input.assign(self.content_img))
            content_layers = [self.vgg.dataDict[layer] for layer in self.content_layers]
            content_layers = sess.run(content_layers)
        gen_content_layers = [self.vgg.dataDict[layer] for layer in self.content_layers]
        self.calculate_content_loss(gen_content_layers, content_layers)
        
        with tf.Session() as sess:
            ### get the style layers of the style image
            sess.run(self.img_input.assign(self.style_img))
            style_layers = [self.vgg.dataDict[layer] for layer in self.style_layers]
            style_layers = sess.run(style_layers)
        gen_style_layers = [self.vgg.dataDict[layer] for layer in self.style_layers]
        self.calculate_style_loss(gen_style_layers, style_layers)
        
        ### calculate total loss based on style weight and content weight
        self.total_loss = self.content_loss_weight*self.style_loss + self.style_loss_weight*self.content_loss 
    
    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss,
                                                                                           global_step=self.global_step)
    
    def build(self):

        self.load_VGG()
        self.calculate_loss()
        self.optimize()
        
    def train(self):
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.img_input.assign(self.gen_img))
            
            initial_step = self.global_step
            start_time = time.time()
            
            for i in range(self.training_epoch):
                sess.run(self.optimizer)
                loss = sess.run(self.total_loss)
                if (i+1) % 1 == 0:
                    print("epoch {}: total_loss = {}".format(i, loss))
                    x = sess.run(self.img_input) + self.vgg.mean_pixels
                    x = np.clip(x,0,255).astype('uint8')
                    #utils.save_image(filename, gen_image)
                    #print(x)
                    cv2.imwrite('./outputs/' + str(i) + '.jpg',cv2.cvtColor(x[0], cv2.COLOR_RGB2BGR))
                    
if __name__ == '__main__':
    machine = Style_Transfer('./content/content9.jpg', './content/style6.jpg', 300, 300)
    machine.build()
    machine.train()


