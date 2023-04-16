import os
# counting the number of files in train folder
path, dirs, files = next(os.walk('/content/train_dataset'))
file_count = len(files)
print('Number of images: ', file_count)
file_names = os.listdir('/content/train_dataset/')
print(file_names)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from google.colab.patches import cv2_imshow
# display Celosa image
img = mpimg.imread('/content/dataset/Celosa/00ab26d4-71a6-4185-970f-e03166ad10ae.jpeg')
imgplt = plt.imshow(img)
plt.show()
# display Crowfoot image
img = mpimg.imread('/content/dataset/Crowfoot grass/03e0ed29-50c9-4872-b174-8d8287b33cfa.jpeg')
imgplt = plt.imshow(img)
plt.show()
# display chloris image
img = mpimg.imread('/content/dataset/Purple Chloris/02d106b4-7015-45d4-b1a0-d269bde487d2.jpeg')
imgplt = plt.imshow(img)
plt.show()
file_names = os.listdir('/content/train_dataset/')

for i in range(5):

  name = file_names[i]
  print(name[0:3])


#creating a directory for resized images
os.mkdir('/content/image reshaped')

original_folder = '/content/train_dataset/'
resized_folder = '/content/image reshaped/'

for i in range(80):

  filename = os.listdir(original_folder)[i]
  img_path = original_folder+filename

  img = Image.open(img_path)
  img = img.resize((224, 224))
  img = img.convert('RGB')

  newImgPath = resized_folder+filename
  img.save(newImgPath)

# display resized  image
img = mpimg.imread('/content/image resized/00ab26d4-71a6-4185-970f-e03166ad10ae.jpeg')
imgplt = plt.imshow(img)
plt.show()

# creaing a for loop to assign labels
filenames = os.listdir('/content/image resized/')


labels = []

for i in range(80):

  file_name = filenames[i]
  label = file_name[0:3]

  if label == 'Ofc':
    labels.append(1)

  else:
    labels.append(0)

  print(filenames[0:5])
  print(len(filenames))

print(labels[0:5])
print(len(labels))

values, counts = np.unique(labels, return_counts=True)
print(values)
print(counts)

import cv2
import glob
image_directory = '/content/image resized/'
image_extension = ['png', 'jpeg']

files = []

[files.extend(glob.glob(image_directory + '*.' + e)) for e in image_extension]

images_weed = np.asarray([cv2.imread(file) for file in files])

print(images_weed)
print(images_weed.shape)
X = images_weed
Y = np.asarray(labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
# scaling the data
X_train_scaled = X_train/255

X_test_scaled = X_test/255

print(X_train_scaled)
import tensorflow as tf
import tensorflow_hub as hub

mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'

pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

model.fit(X_train_scaled, Y_train, epochs=5)
score, acc = model.evaluate(X_test_scaled, Y_test)
print('Test Loss =', score)
print('Test Accuracy =', acc)
input_image_path = input('/content/00ab26d4-71a6-4185-970f-e03166ad10ae.jpeg')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resize = cv2.resize(input_image, (224,224))

input_image_scaled = input_image_resize/255

image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

input_prediction = model.predict(image_reshaped)

print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label == 0:
  print('The image represents a Celosa')
else:
  if input_pred_label == 1:
    print('The image represents a Chloris')
  else:
    print('The given pic is of Crowfoot ')

