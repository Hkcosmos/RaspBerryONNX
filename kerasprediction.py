from keras.models import load_model

model = load_model('C:\\Users\\SM Harikarthik\\Documents\\VSprogramming\\RaspBerryONNX\\model_v1_inceptionV3_new.h5')

import time
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('second.jpg', cv2.IMREAD_ANYCOLOR)
img = cv2.resize(img, (224,224))
x = image.img_to_array(img)
print(x.shape)
plt.imshow(img)



test_img_onnx = x

test_img_input_onnx=np.expand_dims(test_img_onnx,axis=0)
#ground_truth_onnx = np.argmax(y_test_cat[test_img_number_onnx], axis=None)
print(test_img_input_onnx.shape)

test_img_input_onnx=np.array(test_img_input_onnx,dtype=float)

test_img_input_onnx=(test_img_input_onnx.astype('float32'))/255

start_time = time.time()
predictions = model.predict(test_img_input_onnx)
end_time = time.time() - start_time

predicted_class_onnx = np.argmax(predictions, axis=None)

plt.figure(figsize=(2, 2))
plt.axis('off')
plt.imshow(test_img_onnx)

classes = ['burger',
 'butter_naan',
 'chai',
 'chapati',
 'chole_bhature',
 'dal_makhani',
 'dhokla',
 'fried_rice',
 'idli',
 'jalebi',
 'kaathi_rolls',
 'kadai_paneer',
 'kulfi',
 'masala_dosa',
 'momos',
 'paani_puri',
 'pakode',
 'pav_bhaji',
 'pizza',
 'samosa']
#original_label_onnx=classes[ground_truth_onnx]
prediction_label_onnx=classes[predicted_class_onnx]


print("Predicted class using ONNX is:", prediction_label_onnx)
print("Predicted time is: ",end_time)