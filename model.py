
# coding: utf-8

# In[2]:


# import library

import numpy as np
import os, csv, cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.preprocessing.image import random_shear
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Conv2D, Cropping2D, Dropout
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


# Load data
samples =[]
first_line = True
with open('data/udacity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        if first_line :
            first_line = False
        else:    
            samples.append(line)
            
# balance the distribution of data (drop 80% of angles inside the treshold)
            
print(len(samples))            
treshold = 0.1
remove_prob = 0.8
i=0
while i < len(samples):
        if abs(float(samples[i][3])) < treshold:
             if np.random.rand() < remove_prob:
                del samples[i]
        i+=1 
      


# In[5]:


# split data
train_samples, validation_samples = train_test_split(samples,test_size=0.2)

# angle correction
angle_correction = [0, 0.25, -0.25]

# create adjusted steering measurements for the side camera images using a generator
def  generator(samples,batch_size=32):
    num_samples = len(samples)
  
    while 1 :
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
                    
            for batch_sample in batch_samples:
                
                for i, correction in zip(range(3),angle_correction):
                    
                    name = './data/udacity/IMG/' + batch_sample[i].split('/')[-1]
                    image = mpimg.imread(name)
                    angle = float(batch_sample[3]) + correction
                    
                                   
                
                
                    images.append(image)
                    angles.append(angle)
                    
                    
                    
                    #data augumentation: 
                    
                    
                    
                    #flip image
                    images.append(np.fliplr(image))
                    angles.append(-angle)
                    
                    # data agumentation: random shear
                    images.append(random_shear(image, np.random.randint(32)))
                    angles.append(angle)
                    
                   
            # convert to np array
            X_train = np.array(images)
            y_train = np.array(angles)
    
            yield shuffle(X_train,y_train)

                    
                    


# In[6]:


# define generators and image dimensions
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
row, col,ch = 160, 320, 3


# create model
model = Sequential()
# normalize the data and traslate to have zero mean
model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(row,col,ch)))
# define layers
model.add(Cropping2D(cropping=((60,25), (0,0))))


# In[7]:


# architecture layers

model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.7))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(1))

          


# In[8]:


# compile the model          
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer='adam')         


# In[9]:


#train the model

model.fit_generator(train_generator, samples_per_epoch=20000,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)



# Save the model  
model.save('model.h5')


# In[ ]:





# In[ ]:




