import os
import pickle
from PIL import Image
import numpy as np
import cv2 as cv


def face_training():
    trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    recognizer = cv.face.LBPHFaceRecognizer_create()

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
                print(label_ids)
                pil_img = Image.open(path).convert('L')  # Convert to gray
                size = (550, 550)
                final_img = pil_img.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_img, dtype='uint8')

                faces = trained_face_data.detectMultiScale(image_array)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    x_train.append(roi)
                    y_labels.append(id_)
                    # print(roi)
                    # print(h,w)
                    # print(image_array)

                # print(faces)

    # print(x_train)
    # print(y_labels)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('trainner.yml')
