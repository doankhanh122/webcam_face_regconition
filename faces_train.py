import os
from PIL import Image
import numpy as np
import cv2 as cv


trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, 'faces_train')

current_id = 0
label_ids = {}

y_labels = []
x_train = []


for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(' ', '-').lower()
            print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(label_ids)
            pil_img = Image.open(path).convert('L') #Convert to gray
            image_array = np.array(pil_img, dtype='uint8')

            faces = trained_face_data.detectMultiScale(image_array)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                # print(roi)

            # print(faces)


print(x_train)
print(y_labels)