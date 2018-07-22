
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import urllib
import os

### return image from path provided
def get_image(img_addr, img_width, img_height):
    img = cv2.imread(img_addr)
    img = cv2.resize(img, (img_width, img_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape([1, img_width, img_height,3]).astype(np.float64)  
    return img
    
### generate noise image based on original image
### NOTE: the noise_ratio should be in range 0..1
def generate_noise_image(initial_img, img_width, img_height, noise_ratio=0.5):
    noise_img = np.random.uniform(-10,10,size=(1,img_width, img_height, 3)).astype(np.float64)
    assert noise_ratio >= 0
    assert noise_ratio <= 1
    return initial_img*(1-noise_ratio) + noise_img*noise_ratio

def download(url, file_path):
    if os.path.exists(file_path):
        print("the file is already existed")
        return
    else:
        print("downloading file...")
    urllib.request.urlretrieve(url, file_path) 
    print("downloading done")

