import cv2 as cv
import pickle
import numpy as np

# TODO Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# # Eye Dectection cascade
# trained_eye_data = cv.CascadeClassifier('haarcascade_eye.xml')

#
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {"person_name": 1}
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# TODO Choose an image to detect face in
# img = cv.imread('photos/khanh.JPG')

# TODO Capture video from webcam
webcam = cv.VideoCapture(0)
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()
    # Must convert to Grayscale
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Coordinates of face detected
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame, scaleFactor=1.5, minNeighbors=6)
    # eye_coordinates = trained_eye_data.detectMultiScale(grayscale_frame, 1.5, 6)

    # Detect and Draw rectangle to face
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        # TODO Recognition
        roi_gray = grayscale_frame[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(roi_gray)
        if confidence >= 45:
            print(id_)
            print(labels[id_])
            print(confidence)
            # Display text
            font = cv.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 255, 0)
            stroke = 2
            cv.putText(frame, name, (x, y), fontFace=font, fontScale=1, color = color, thickness=stroke, lineType=cv.LINE_AA)

    # # Detect and Draw Eyes\
    # for (x, y, w, h) in eye_coordinates:
    #     cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


    cv.imshow('My webcam', frame)
    key = cv.waitKey(1)


    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break


# TODO Must convert to grayscale
grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# TODO Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# TODO Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


# TODO Display image with the face
cv.imshow('My face', img)
#
cv.waitKey()


print('Code completed')


