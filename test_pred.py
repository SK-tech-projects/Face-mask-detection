from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json
import tensorflow as tf
import cv2

img_width, img_height = 224,224
path=r"F:/Face-Mask-Detection/mask_detector"

model = load_model(path)

def net(inputImage):
    img = image.load_img(inputImage, target_size=(img_width, img_height))
    x = np.array(img)
    # print(x)
    # print(x.shape)
    print(x[0].shape)
    x = np.expand_dims(x, axis=0)
    # print(x.shape)
    # print(x[0].shape)
    prediction = model.predict(x)
    # print(prediction)

    score = prediction[0][0]
    res="Person is {} percent without mask and {} percent masked.".format(100 * (1 - score), 100 * score)
    return res

ans = net('test.jpg')
print(ans)
ans = net('test2.jpg')
print(ans)
ans = net('test3.jpg')
print(ans)
ans = net('C:/Users/kunal/Desktop/KUNAL/kunal_photo.jpg')
print(ans)
ans = net('test4.jpg')
print(ans)