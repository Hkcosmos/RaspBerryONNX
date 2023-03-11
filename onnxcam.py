import os
import cv2
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='/home/root34/archiconda3/envs/envname/RaspBerryONNX/Download.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('/home/root34/archiconda3/envs/envname/RaspBerryONNX/Download.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('/home/root34/archiconda3/envs/envname/RaspBerryONNX/Download.jpg', cv2.IMREAD_ANYCOLOR)
            print("Converting RGB image to grayscale...")
            gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            print("Converted RGB image to grayscale...")
            print("Resizing image to 28x28 scale...")
            img_ = cv2.resize(gray,(28,28))
            print("Resized...")
            img_resized = cv2.imwrite(filename='/home/root34/archiconda3/envs/envname/RaspBerryONNX/saved_img.jpg', img=img_)
            print("Image saved!")
        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('/home/root34/archiconda3/envs/envname/RaspBerryONNX/Download.jpg', cv2.IMREAD_ANYCOLOR)
img = cv2.resize(img, (224,224))
x = image.img_to_array(img)
print(x.shape)
#plt.imshow(img)
################################
#Inference using ONNX (and keras for comparison)

import onnxruntime
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.losses import *

session = onnxruntime.InferenceSession("InceptionV3_new_model.onnx")
session.get_inputs()[0].shape
session.get_inputs()[0].type

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

test_img_onnx = x

test_img_input_onnx=np.expand_dims(test_img_onnx,axis=0)
#ground_truth_onnx = np.argmax(y_test_cat[test_img_number_onnx], axis=None)
print(test_img_input_onnx.shape)
print(output_name)
print(input_name)
test_img_input_onnx=np.array(test_img_input_onnx,dtype=float)
test_img_input_onnx=(test_img_input_onnx.astype('float32'))/255
result = session.run([output_name], {input_name: test_img_input_onnx})
predicted_class_onnx = np.argmax(result, axis=None)

plt.figure(figsize=(2, 2))
plt.axis('off')
#plt.imshow(test_img_onnx)

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