import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

# Initialize CNN

classifier = Sequential()
classifier.add(Convolution2D(32, 3,3 , input_shape = (64,64 ,3) , activation = 'relu'))
classifier.add(MaxPooling2D(pool_size =(2,2)))
classifier.add(Flatten())

# Full connection

classifier.add(Dense(output_dim =128 , activation = 'relu'))
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss ='binary_crossentropy' ,metrics =['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale =1./255,
                                  shear_range =0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set =train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64) , batch_size= 32 , class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size =(64,64) , batch_size= 32 , class_mode= 'binary')

from IPython.display import display
import PIL
import sys
from PIL import Image
sys.modules['Image'] = Image 
from sklearn.preprocessing import LabelEncoder

from PIL import Image
print(Image.__file__)

import Image
print(Image.__file__)

classifier.fit_generator(training_set , steps_per_epoch= 8000 , epochs= 5 , validation_data=test_set, validation_steps= 20)

# Testing Results on random downloaded image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("puppy-1903313__340.jpg",target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image , axis= 0)
result =classifier.predict(test_image)
print(result)

# chacking the labels associated with the images
training_set.class_indices

if result[0][0] >=0.5 :
    predictions = 'Dog'
else :
    predictions = 'Cat'
    
print(predictions)
